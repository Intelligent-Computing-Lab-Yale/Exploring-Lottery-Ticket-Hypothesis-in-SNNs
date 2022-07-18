import utils
import config_lth
import time
import torchvision
import os
import copy

from archs.cifarsvhn.vgg import vgg16_bn
from archs.cifarsvhn.resnet import ResNet19
from archs.fmnist.vgg import vgg16_bn as fmnist_vgg16_bn
from archs.fmnist.resnet import ResNet19 as fmnist_ResNet19

from utils_for_snn_lth import *
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net



def main():
    args = config_lth.get_args()


    # define dataset
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'fmnist':
        trainset = torchvision.datasets.FashionMNIST(root=os.path.join(args.data_dir, 'fmnist'), train=True,
                                                     download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.FashionMNIST(root=os.path.join(args.data_dir, 'fmnist'), train=False,
                                                   download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10
    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=os.path.join(args.data_dir, 'svhn'), split='train',
                                             download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.SVHN(root=os.path.join(args.data_dir, 'svhn'), split='test',
                                           download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10
    elif args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class =100


    criterion = nn.CrossEntropyLoss()

    if args.dataset != 'fmnist' and args.arch == 'vgg16':
        model = vgg16_bn(num_classes=n_class, total_timestep=args.timestep).cuda()
    elif args.dataset != 'fmnist' and args.arch == 'resnet19':
        model = ResNet19(num_classes=n_class, total_timestep=args.timestep).cuda()
    elif args.dataset == 'fmnist' and args.arch == 'vgg16':
        model = fmnist_vgg16_bn(num_classes=n_class, total_timestep=args.timestep).cuda()
    elif args.dataset == 'fmnist' and args.arch == 'resnet19':
        model = fmnist_ResNet19(num_classes=n_class, total_timestep=args.timestep).cuda()
    else:
        exit()


    # Copying and Saving Initial State
    original_model = copy.deepcopy(model)
    initial_state_dict = copy.deepcopy(model.state_dict())

    # Making Initial Mask
    mask = make_mask(model)


    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate*2, args.momentum, args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate*2)
    else:
        print ("will be added...")
        exit()

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.end_iter*0.5),int(args.end_iter*0.75)], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.end_iter), eta_min= 0)
    else:
        print ("will be added...")
        exit()

    best_accuracy = 0

    sparsity_list = [0,	25.04,	43.71,	57.76,	68.3, 76.2,	82.13, 86.58, 89.91, 92.41,	94.29, 95.69, 96.75, 97.54,	98.13]
    args.prune_percent = sparsity_list[args.sparsity_round]

    # TODO [ealry-brid ticket] - finding EB in one ticket
    mask_list = []
    epoch_keep = 5
    dist_list = [1] * (epoch_keep-1)
    early_bird_ticket = 0


    for epoch in range(args.end_iter):

        mask = get_pruning_maks(args, args.prune_percent, mask, model)
        early_bird_ticket = copy.deepcopy(mask)

        # TODO [put mask into queue]
        if len(mask_list) < epoch_keep:
            mask_list.append(copy.deepcopy(mask))
        else:
            mask_list.pop(0)
            mask_list.append(copy.deepcopy(mask))

        # TODO [put mask into queue]

        if len(mask_list) == epoch_keep:
            for i in range(len(mask_list) - 1):
                mask_i = mask_list[-1]
                mask_j = mask_list[i]

                numerator = 0
                denominator = 0

                for ly in range(len(mask_i)):
                    numerator += torch.sum(torch.Tensor(mask_i[ly]) == torch.Tensor(mask_j[ly]))
                    denominator += torch.prod(torch.Tensor(mask_j[ly].shape))
                dist_list[i] = 1 - float(numerator) / denominator
            # print (dist_list)
            if max(dist_list) < 0.02:
                print ("find early bird ticket at epoch", epoch)
                break


        # #Training
        loss = train_normal(args, epoch, train_loader, model, criterion, optimizer, scheduler)

        # Frequency for Testing
        if (epoch+1) % args.valid_freq == 0:
            accuracy = test(model, val_loader, criterion)
            print('[Val_Accuracy epoch:%d] val_acc:%f'
                  % (epoch + 1, accuracy))
            # # Save Weights
            if accuracy > best_accuracy:
                best_accuracy = accuracy


        # Frequency for Printing Accuracy and Loss
        if (epoch+1) % args.print_freq == 0:
            print(
                f'Train Epoch: {epoch}/{args.end_iter} Loss: {loss:.6f} % Best Accuracy: {best_accuracy:.2f}%')


    # TODO [ealry-brid ticket] - phase2: retraining founded EB ticket
    print (args.prune_percent)

    model = original_initialization(early_bird_ticket, initial_state_dict, original_model)
    comp1 = utils.print_nonzeros(model)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        print ("will be added...")
        exit()

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.end_iter*0.5),int(args.end_iter*0.75)], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.end_iter), eta_min= 0)
    else:
        print ("will be added...")
        exit()




    start = time.time()
    for epoch in range(args.end_iter):
        train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
        scheduler.step()

        if (epoch + 1) % args.valid_freq == 0:
            accuracy = test(model, val_loader, criterion)
            print('[Val_Accuracy epoch:%d] val_acc:%f'
                  % (epoch + 1, accuracy))


def train_normal(args, epoch, train_data,  model, criterion, optimizer, scheduler):
    model.train()
    EPS = 1e-6

    for batch_idx, (imgs, targets) in enumerate(train_data):

        train_loss = 0.0

        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()
        output_list = model(imgs)
        for output in output_list:
            train_loss += criterion(output, targets) / args.timestep
        train_loss.backward()
        optimizer.step()
        reset_net(model)

    scheduler.step()

    return train_loss.item()



def train(args, epoch, train_data,  model, criterion, optimizer, scheduler):
    model.train()
    EPS = 1e-6

    for batch_idx, (imgs, targets) in enumerate(train_data):

        train_loss = 0.0

        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()
        output_list = model(imgs)
        for output in output_list:
            train_loss += criterion(output, targets) / args.timestep

        train_loss.backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data
                if (len(tensor.size())) == 1:
                    continue
                grad_tensor = p.grad
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor
        optimizer.step()
        reset_net(model)
    scheduler.step()


    return train_loss.item()






if __name__ == '__main__':
    main()
