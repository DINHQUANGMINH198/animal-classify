import os
import time
import copy
import torch
import pandas as pd
from torchvision import models

import cfg
from utils import fprint, calculate_metrics, get_sub_dump_dir,load_args
def initialize_model(is_pretrained):

    model = models.alexnet(pretrained=is_pretrained)

    # initially disable all parameter updates
    if is_pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # reshape the output layer
    in_size = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_size, cfg.NUM_CATEGORIES)

    if is_pretrained:
        params_to_update = []
        for param in model.parameters():
            if param.requires_grad:
                params_to_update.append(param)  # parameters of reshaped layer
    else:
        params_to_update = model.parameters()  # parameters of all layers

    return model, params_to_update

#
#
#
#
#
#


def train_model(model, data_loaders, criterion, optimizer, args):
    args = load_args()

    # create states df and csv file
    stats_df = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1'])

    sub_dump_dir = get_sub_dump_dir(args)
    stats_path = os.path.join(sub_dump_dir, 'stats.csv')

    stats_df.to_csv(stats_path, sep=',', index=False)  # write loss and acc values
    fprint('\nCreated stats file\t-> {}'.format(stats_path), args)
    fprint('\nTRAINING {} EPOCHS...\n'.format(args.epochs), args)

    since = time.time()

    # initialize best values
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_opt_state_dict = copy.deepcopy(optimizer.state_dict())
    best_loss = 999999.9
    best_acc = 0.0
    best_epoch = 0
    if args.checkpoint != "" :
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    for epoch in range(args.epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            phase_loss = 0.0
            phase_corrects = 0
            phase_preds = torch.LongTensor().to(torch.device(args.device))
            phase_category_ids = torch.LongTensor().to(torch.device(args.device))

            # Iterate over data
            for inputs, category_ids in data_loaders[phase]:
                inputs = inputs.to(torch.device(args.device))
                category_ids = category_ids.to(torch.device(args.device))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, category_ids)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # stats
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == category_ids.data)
                phase_loss += batch_loss
                phase_corrects += batch_corrects
                phase_preds = torch.cat((phase_preds, preds), 0)
                phase_category_ids = torch.cat((phase_category_ids, category_ids), 0)

            epoch_loss = phase_loss / len(data_loaders[phase].dataset)
            epoch_acc, epoch_f1 = calculate_metrics(phase_preds, phase_category_ids)

            stats_df.at[0, 'epoch'] = epoch
            stats_df.at[0, phase + '_loss'] = round(epoch_loss, 6)
            stats_df.at[0, phase + '_acc'] = round(epoch_acc, 6)
            stats_df.at[0, phase + '_f1'] = round(epoch_f1, 6)

            # define the new bests
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_opt_state_dict = copy.deepcopy(optimizer.state_dict())
                best_loss = copy.deepcopy(epoch_loss)
                best_epoch = epoch

        # append epoch stats to file
        fprint(stats_df.to_string(index=False, header=(epoch == 0), col_space=10, justify='right'), args)
        stats_df.to_csv(stats_path, mode='a', header=False, index=False)

    time_elapsed = time.time() - since
    fprint('\nTraining completed in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60), args)

    # reload best model weights and best optimizer variables
    model.load_state_dict(best_model_state_dict)
    optimizer.load_state_dict(best_opt_state_dict)

    # save best checkpoint
    if not os.path.exists(cfg.MODEL_DIR):
        os.makedirs(cfg.MODEL_DIR)

    cp_path = os.path.join(cfg.MODEL_DIR, '{}_{}_{:.6f}.pth'.format(
        'pt' if args.pretrained else 'fs', args.t_start, best_acc))

    if args.save:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state_dict,
            'optimizer_state_dict': best_opt_state_dict,
            'loss': best_loss,
            'acc': best_acc
        }, cp_path)
        fprint('Saved best checkpoint\t-> {}'.format(cp_path), args)

    return model, optimizer

#
#
#
#
#
#


def test_model(model, data_loaders, args):
    fprint('\nTESTING...', args)
    was_training = model.training  # store mode
    model.eval()  # run in evaluation mode

    with torch.no_grad():
        phase_corrects = 0
        phase_preds = torch.LongTensor()
        phase_category_ids = torch.LongTensor()

        for inputs, category_ids in data_loaders['test']:
            inputs = inputs.to(torch.device(args.device))
            category_ids = category_ids.to(torch.device(args.device))

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            batch_corrects = torch.sum(preds == category_ids.data)
            phase_corrects += batch_corrects
            phase_preds = torch.cat((phase_preds, preds), 0)
            phase_category_ids = torch.cat((phase_category_ids, category_ids), 0)

        dataset = data_loaders['test'].dataset
        acc, f1 = calculate_metrics(phase_preds, phase_category_ids)

        fprint('{}/{} predictions are correct -> Test acc: {:.6f}   f1: {:.6f}\n'.format(
            phase_corrects, len(dataset), acc, f1), args)

    model.train(mode=was_training)  # reinstate the previous mode

    return acc
