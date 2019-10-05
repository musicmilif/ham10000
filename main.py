import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import matplotlib
matplotlib.use('agg')

from src.utils import generate_meta, stratified_split, create_loss_plot, setup_logging
from src.data_utils import HAMDataset, build_train_transform, build_test_transform, build_preprocess
from modeling.model import HAMNet
from modeling.utils import save_checkpoint, AverageMeter

STATUS_MSG_T = "Batches done: {}/{} | Loss: {:04f} | Accuracy: {:04f}"
STATUS_MSG_V = "Epochs done: {}/{} | Loss: {:04f} | Accuracy: {:04f}"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Logging
    LOGGER = logging.getLogger(__name__)
    exp_dir = os.path.join('experiments', '{}'.format(args.exp_name))
    log_file = os.path.join(exp_dir, 'log.log')
    npy_file = os.path.join(exp_dir, 'final_results.npy')
    os.makedirs(exp_dir, exist_ok=True)
    setup_logging(log_path=log_file, log_level=args.log_level, logger=LOGGER)
    args_file = os.path.join(exp_dir, 'args.log')
    with open(args_file, 'w') as f:
        f.write(str(args))

    # Initialize datasets and loaders.
    LOGGER.info('Data Processing...')

    df = generate_meta(args.data_dir)
    train_ids, valid_ids = stratified_split(df)
    n_classes = df['target'].max()+1

    model = HAMNet(n_classes, model_name=args.backbone)
    model = model.to(device)

    train_dataset = HAMDataset(
        train_ids,
        df,
        build_preprocess(model.mean, model.std),
        build_train_transform()
        )
    valid_dataset = HAMDataset(
        valid_ids,
        df,
        build_preprocess(model.mean, model.std),
        build_test_transform()
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # sampler=RandomSampler(train_dataset),
        num_workers=8
        )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size,
        num_workers=8
        )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0
    epochs, train_losses, valid_losses = [], [], []
    for epoch in range(1, args.num_epochs+1):
        # Training
        LOGGER.info(f'Epoch: {epoch}')
        model.train()
        n_batches = len(train_loader.dataset) // args.batch_size + 1

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(dim=1)

            train_loss.update(loss.item())
            train_acc.update(predicted.eq(targets).sum().item()/targets.size(0))

            LOGGER.info(STATUS_MSG_T.format(batch_idx+1,
                                            n_batches,
                                            train_loss.avg,
                                            train_acc.avg))
        # Validation
        model.eval()

        valid_loss = AverageMeter()
        valid_acc = AverageMeter()

        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(valid_loader):
                inputs, targets = inputs.to(device), targets.type(torch.LongTensor).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                _, predicted = outputs.max(dim=1)
                valid_loss.update(loss.item())
                valid_acc.update(predicted.eq(targets).sum().item()/targets.size(0))

        LOGGER.info(STATUS_MSG_V.format(epoch,
                                        args.num_epochs,
                                        valid_loss.avg,
                                        valid_acc.avg))

        # Save checkpoint.
        if valid_acc.avg > best_acc:
            LOGGER.info('Saving..')
            output_file_name = os.path.join(exp_dir, f'checkpoint_{valid_acc.avg:.3f}.ckpt')
            save_checkpoint(path=output_file_name,
                            model=model,
                            epoch=epoch,
                            optimizer=optimizer)
            best_acc = valid_acc.avg

        epochs.append(epoch)
        train_losses.append(train_loss.avg)
        valid_losses.append(valid_loss.avg)
        create_loss_plot(exp_dir, epochs, train_losses, valid_losses)
        np.save(npy_file, [train_losses, valid_losses])


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='PyTorch classifier on HAM10000 dataset')
    parser.add_argument('--data-dir', default='/disk/HAM10000/', help='path to data')
    parser.add_argument('--exp-name', default='baseline', type=str,
                        help='name of experiment')
    parser.add_argument('--log-level', default='INFO', choices = ['DEBUG', 'INFO'],
                        help='log-level to use')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='batch-size to use')
    parser.add_argument('--backbone', default='resnet18', choices=['resnet18', 'se_resnet50'],
                        help='network architecture')
    parser.add_argument('--num-epochs', default=10, type=int,
                        help='Number of training epochs')
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))