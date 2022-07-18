import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from spikingjelly.clock_driven.functional import reset_net


def make_mask(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

    return mask

# Prune by Percentile module
def prune_by_percentile(args, percent, mask , model):


        if args.pruning_scope == 'local':
            # Calculate percentile value
            step = 0
            for name, param in model.named_parameters():

                # We do not prune bias term
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    if (len(tensor.shape)) == 1:
                        step += 1
                        continue
                    alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                    percentile_value = np.percentile(abs(alive), percent)

                    # Convert Tensors to numpy and calculate
                    weight_dev = param.device
                    new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                    # Apply new weight and mask
                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    mask[step] = new_mask
                    step += 1
            step = 0
        elif args.pruning_scope == 'global':
            step = 0
            all_param = []
            for name, param in model.named_parameters():
                # We do not prune bias term
                if 'weight' in name:
                    tensor = param.data.cpu().numpy()
                    if (len(tensor.shape)) == 1: # We do not prune BN term
                        continue
                    alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
                    all_param.append(list(abs(alive)))
            param_whole = np.concatenate(all_param)
            percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0])/float(100./percent))]

            step = 0

            for name, param in model.named_parameters():
                # We do not prune bias term
                if 'weight' in name:
                    tensor =  param.data.cpu().numpy()
                    if (len(tensor.shape)) == 1:  # We do not prune BN term
                        step += 1
                        continue

                    # Convert Tensors to numpy and calculate
                    weight_dev = param.device
                    new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                    # Apply new weight and mask
                    param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                    mask[step] = new_mask
                    step += 1
            step = 0
        else:
            exit()

        return model, mask


def get_pruning_maks(args, percent, mask, model):
    step = 0
    all_param = []
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if (len(tensor.shape)) == 1:  # We do not prune BN term
                continue
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
            all_param.append(list(abs(alive)))
    param_whole = np.concatenate(all_param)
    percentile_value = np.sort(param_whole)[int(float(param_whole.shape[0]) / float(100. / percent))]

    step = 0

    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            if (len(tensor.shape)) == 1:  # We do not prune BN term
                step += 1
                continue
            new_mask = np.where(abs(tensor) < percentile_value, 0, torch.FloatTensor([1]))
            mask[step] = new_mask
            step += 1
    step = 0

    return  mask


def original_initialization(mask_temp, initial_state_dict, model):

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

    return model

def original_initialization_nobias(mask_temp, initial_state_dict, model):

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name] +1

    step = 0

    return model


# Function for Testing
def test(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = sum(output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def test_ann(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            reset_net(model)

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy




def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)

