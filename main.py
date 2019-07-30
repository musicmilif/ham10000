"""Train classifier on HAM10000 dataset.
"""

import argparse
import matplotlib
matplotlib.use('agg')
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
import torch.backends.cudnn as cudnn

from networks.SimpleCNN import *
from lib.dataset import *
from lib.utils import *


# Parse args.
parser = argparse.ArgumentParser(description='PyTorch classifier on HAM10000 dataset')
parser.add_argument('--data-dir', default='./data', help='path to data')
parser.add_argument('--train_fraction', default=0.01, type=float,
                    help='fraction of dataset to use for training')
parser.add_argument('--val_fraction', default=0.01, type=float,
                    help='fraction of dataset to use for validation')
parser.add_argument('--exp-name', default='baseline', type=str,
                    help='name of experiment')
parser.add_argument('--log-level', default='INFO', choices = ['DEBUG', 'INFO'],
                    help='log-level to use')
parser.add_argument('--batch-size', default=4, type=int,
                    help='batch-size to use')
parser.add_argument('--network', default='SimpleCNN', choices=['SimpleCNN'],
                    help='network architecture')
parser.add_argument('--num-epochs', default=10, type=int,
                    help='Number of training epochs')
args = parser.parse_args()


# Globals.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Logging.
LOGGER = logging.getLogger(__name__)
exp_dir = os.path.join('experiments', '{}'.format(args.exp_name))
log_file = os.path.join(exp_dir, 'log.log')
npy_file = os.path.join(exp_dir, 'final_results.npy')
os.makedirs(exp_dir, exist_ok=True)
setup_logging(log_path=log_file, log_level=args.log_level, logger=LOGGER)
args_file = os.path.join(exp_dir, 'args.log')
with open(args_file, 'w') as the_file:
    the_file.write(str(args))
STATUS_MSG = "Batches done: {}/{} | Loss: {:04f} | Accuracy: {:04f}"


# Initialize datasets and loaders.
LOGGER.info('==> Preparing data..')
train_ids, val_ids = create_train_val_split(args.data_dir,
                                            args.train_fraction,
                                            args.val_fraction)
train_set = HAM10000(args.data_dir, train_ids)
val_set = HAM10000(args.data_dir, val_ids)

train_sampler = RandomSampler(train_set)
num_classes = train_set.get_num_classes()
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=args.batch_size,
                                           sampler=train_sampler,
                                           num_workers=8)
val_loader = torch.utils.data.DataLoader(val_set,
                                         batch_size=args.batch_size,
                                         num_workers=8)


# Model.
LOGGER.info('==> Building model..')
if args.network == 'SimpleCNN':
    net = SimpleCNN(num_classes=num_classes)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


# Training.
def train(epoch):
    LOGGER.info('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    n_batches = len(train_loader.dataset) // args.batch_size

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        LOGGER.info(STATUS_MSG.format(batch_idx+1,
                                      n_batches,
                                      train_loss/(batch_idx+1),
                                      100.*correct/total))

    return train_loss/(batch_idx+1)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    n_batches = len(val_loader.dataset) // args.batch_size

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            LOGGER.info(STATUS_MSG.format(batch_idx+1,
                                          n_batches,
                                          test_loss/(batch_idx+1),
                                          100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if acc > best_acc:
        LOGGER.info('Saving..')
        save_checkpoint(state, exp_dir, backup_as_best=True)
        best_acc = acc
    else:
        save_checkpoint(state, exp_dir, backup_as_best=False)

    return test_loss/(batch_idx+1)


if __name__ == '__main__':
    epochs, train_losses, test_losses = [], [], []

    for epoch in range(0, args.num_epochs):
        train_loss = train(epoch)
        test_loss = test(epoch)
        epochs.append(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        create_loss_plot(exp_dir, epochs, train_losses, test_losses)
        np.save(npy_file, [train_losses, test_losses])
