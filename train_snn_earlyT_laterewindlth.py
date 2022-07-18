import time
import utils
import config_lth
import torchvision
import os
import pickle
import copy

from archs.cifarsvhn.vgg import vgg16_bn
from archs.cifarsvhn.resnet import ResNet19
from archs.fmnist.vgg import vgg16_bn as fmnist_vgg16_bn
from archs.fmnist.resnet import ResNet19 as fmnist_ResNet19
from utils_for_snn_lth import *
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net


def find_ET_ticket(args, model, trainset,epoch):
    with torch.no_grad():

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1000,
                                                   shuffle=True, pin_memory=True, num_workers=4)

        # Grab a single batch from the training dataset
        inputs, targets = next(iter(train_loader))
        inputs = inputs.cuda()
        targets = targets.cuda()


        output_list = model(inputs)
        out_prob_t5 = sum(output_list) / args.timestep
        reset_net(model)

        candidate_t = [2,3,4]
        KDdivs = []
        for temp_t in candidate_t:
            model.total_timestep = temp_t
            output_list = model(inputs)
            out_prob_t_temp = sum(output_list) / temp_t

            KDdiv = nn.KLDivLoss()(F.log_softmax(out_prob_t_temp, dim=1),
                                     F.softmax(out_prob_t5, dim=1))
            KDdivs.append(float(KDdiv.cpu().data.numpy())*1e4)

            reset_net(model)

        max_KLdivs = max(KDdivs)
        norm_KLdivs = []
        threshold = 0.6
        new_timestep = 5
        for i, kldiv in enumerate(KDdivs):
            norm_KLdiv = kldiv/max_KLdivs
            norm_KLdivs.append(norm_KLdiv)

        for i, n_kldiv in enumerate(norm_KLdivs):
            if n_kldiv < threshold:
                new_timestep = candidate_t[i]
                break


    print ('Find earlybrid timestep-------')
    print ('epoch', epoch, ':', norm_KLdivs, '|Newtimesetp', new_timestep)

    return  new_timestep



def main():
    s_time = time.time()
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
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/ETsnn_laterewind_lth/{args.arch}/{args.dataset}/round{args.round}")
    torch.save(model.state_dict(), f"{os.getcwd()}/ETsnn_laterewind_lth/{args.arch}/{args.dataset}/round{args.round}/initial_state_dict.pth.tar")

    # Making Initial Mask
    mask = make_mask(model)


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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.end_iter), eta_min= args.learning_rate*0.01)
    else:
        print ("will be added...")
        exit()

    # Pruning

    best_accuracy = 0
    ITERATION = args.prune_iterations
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)

    rewinding_epoch = 0

    for _ite in range(ITERATION):
        if not _ite == 0:

            model, mask = prune_by_percentile(args, args.prune_percent, mask , model)
            model = original_initialization(mask, initial_state_dict, model)

            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
            else:
                exit()
            if args.scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.end_iter * 0.5),
                                                                                        int(args.end_iter * 0.75)],
                                                                 gamma=0.1)
            elif args.scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.end_iter- rewinding_epoch),
                                                                       eta_min= 0)
            else:
                exit()


            #TODO bringback T for the last train
            if _ite == (ITERATION-1) :
                print("search_time... ", time.time() - s_time)
                print ("recover original timestep", args.timestep)
                model.total_timestep = args.timestep

        print(f"\n--- Pruning Level [round{args.round}:{_ite}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[_ite] = comp1

        for iter_ in range(args.end_iter - rewinding_epoch):
            # Frequency for Testing
            if (iter_+1) % args.valid_freq == 0:
                accuracy = test(model, val_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/snn_ETlaterewind_lth/{args.arch}/{args.dataset}/round{args.round}")
                    torch.save(model,
                               f"{os.getcwd()}/snn_ETlaterewind_lth/{args.arch}/{args.dataset}/round{args.round}/{_ite}_model.pth.tar")

            # Training
            loss = train(args, iter_, train_loader, model, criterion, optimizer, scheduler)


            #TODO 1. find_ET_ticket 2.Late rewinding init weight at 20epoch
            if _ite == 0 and iter_ == 1:
                new_timestep = find_ET_ticket(args, model, trainset, iter_)
                model.total_timestep = new_timestep

            if _ite == 0 and iter_ == args.rewinding_epoch:
                print ('find laterewinding weight--------')
                initial_state_dict = copy.deepcopy(model.state_dict())
                torch.save(initial_state_dict,
                           f"{os.getcwd()}/ETsnn_laterewind_lth/{args.arch}/{args.dataset}/round{args.round}/initial_state_dict_rewind.pth.tar")

                rewinding_epoch = args.rewinding_epoch

            # Frequency for Printing Accuracy and Loss
            if (iter_+1) % args.print_freq == 0:
                print(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')


        bestacc[_ite]=best_accuracy

        # Dumping mask
        utils.checkdir(f"{os.getcwd()}/dumps/ETsnn_laterewind_lth/{args.arch}/{args.dataset}/round{args.round}")
        with open(f"{os.getcwd()}/dumps/ETsnn_laterewind_lth/{args.arch}/{args.dataset}/round{args.round}/mask_{comp1}.pkl",
                  'wb') as fp:
            pickle.dump(mask, fp)

        # Making variables into 0
        best_accuracy = 0




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
