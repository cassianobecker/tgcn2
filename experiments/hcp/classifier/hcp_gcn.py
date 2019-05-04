from __future__ import print_function
import argparse
import random
import os

import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

from dataset.hcp.torch_data import loaders
from nn.chebnet import ChebTimeConv


class NetTGCNBasic(torch.nn.Module):
    """
    A 1-Layer time graph convolutional network
    :param mat_size: temporary parameter to fix the FC1 size
    """
    def __init__(self, mat_size):
        super(NetTGCNBasic, self).__init__()

        f1, g1, k1, h1 = 1, 64, 25, 15
        self.conv1 = ChebTimeConv(f1, g1, K=k1, H=h1)

        n2 = mat_size
        c = 6
        self.fc1 = torch.nn.Linear(int(n2 * g1), c)

        self.coos = None
        self.perm = None

    def add_graph(self, coos, perm):
        """
        Sets the COO adjacency matrix (or matrices post-coarsening) for the graph and the order of vertices in the matrix
        :param coos: list of adjacency matrices for the graph
        :param perm: order of vertices in the adjacency matrix
        :return: None
        """
        self.coos = coos
        self.perm = perm

    def forward(self, x):
        """
        Computes forward pass through the time graph convolutional network
        :param x: windowed BOLD signal to as input to the TGCN
        :return: output of the TGCN forward pass
        """
        x, edge_index = x, self.coos[0]

        x = self.conv1(x, edge_index)

        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


def train_minibatch(args, model, device, train_loader, optimizer, epoch, mini_batch=10, verbose=True):
    """
    Loads input data (BOLD signal windows and corresponding target motor tasks) from one patient at a time,
    and minibatches the windowed input signal while training the TGCN by optimizing for minimal training loss.
    :param args: keyword arguments (see main())
    :param model: PyTorch Module/DataParallel object to model
    :param device: device to send the data to
    :param train_loader: DataLoader that hosts the training data
    :param optimizer: optimizing algorithm (default=Adam)
    :param epoch: current epoch
    :param mini_batch: number of minibatches to go through before taking an optimizer step
    :param verbose: boolean to print out training progress
    :return: train_loss
    """
    train_loss = 0
    model.train()

    for batch_idx, (data, target, coos, perm) in enumerate(train_loader):

        coos = [c[0].to(device) for c in coos]
        target = target.to(device)
        temp_loss = 0
        model.module.add_graph(coos, perm)

        for i in range(len(data)):

            optimizer.zero_grad()
            output = model(data[i].to(device))
            expected = torch.argmax(target[:, i], dim=1)

            k = 1.
            w = torch.tensor([1., k, k, k, k, k]).to(device)

            loss = F.nll_loss(output, expected, weight=w)
            loss = loss / mini_batch
            train_loss += loss
            temp_loss += loss

            for p in model.named_parameters():
                if p[0].split('.')[0][:2] == 'fc':
                    loss = loss + args.reg_weight * (p[1] ** 2).sum()

            loss.backward()

            if batch_idx % mini_batch == 0:
                optimizer.step()

        if verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx + 1, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader.dataset),
                temp_loss.item()))

    train_loss /= (len(train_loader.dataset) * len(data))
    return train_loss


def test(args, model, device, test_loader, epoch, verbose=True):
    """
    Evaluates the model trained in train_minibatch() on patients loaded from the test set.
    :param args: keyword arguments (see main())
    :param model: PyTorch Module/DataParallel object to model
    :param device: device to send the data to
    :param test_loader: DataLoader that hosts the test data
    :param epoch: current epoch
    :param verbose: boolean to print out test progress
    :return: test_loss and correct, the # of correct predictions
    """
    model.eval()

    test_loss = 0
    correct = 0

    preds = torch.empty(0, dtype=torch.long).to(device)
    targets = torch.empty(0, dtype=torch.long).to(device)

    with torch.no_grad():
        for batch_idx, (data_t, target_t, coos, perm) in enumerate(test_loader):

            coos = [c[0].to(device) for c in coos]
            # data = data_t[0].to(device)
            target = target_t.to(device)

            model.module.add_graph(coos, perm)

            for i in range(len(data_t)):
                output = model(data_t[i].to(device))
                expected = torch.argmax(target[:, i], dim=1)
                test_loss += F.nll_loss(output, expected, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                preds = torch.cat((pred, preds))
                targets = torch.cat((expected, targets))
                correct += pred.eq(expected.view_as(pred)).sum().item()

    test_loss /= (len(test_loader.dataset) * len(data_t))

    if verbose:
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx + 1, len(test_loader.dataset),
                   100. * (batch_idx + 1) / len(test_loader.dataset),
            test_loss.item()))

    return test_loss, correct


def experiment(args):
    """
    Sets up the experiment environment (loggers, data loaders, model, optimizer and scheduler) and initiates the
    train/test process for the model.
    :param args: keyword arguments from main() as parameters for the experiment
    :return: None
    """

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # coarsening_levels = 4

    batch_size = 1
    train_loader, test_loader, mat_size = loaders(device, batch_size=batch_size, download_train=True, download_test=True)

    model = NetTGCNBasic(mat_size)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(1, args.epochs + 1):

        train_loss = train_minibatch(args, model, device, train_loader, optimizer, epoch, mini_batch=batch_size,
                                     verbose=True)

        scheduler.step()

        test_loss, correct = test(args, model, device, test_loader, epoch)

        print('Epoch: {} Training loss: {:1.3e}, Test loss: {:1.3e}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch, train_loss, test_loss, correct, len(test_loader.dataset) * 270,
                                                   100. * correct / (len(test_loader.dataset) * 270)))

    if args.save_model:
        torch.save(model.state_dict(), "hcp_cnn_1gpu2.pt")


def seed_everything(seed=1234):
    """
    Sets a random seed for OS, NumPy, PyTorch and CUDA.
    :param seed: random seed to apply
    :return: None
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    args.reg_weight = 5.e-4

    torch.manual_seed(args.seed)

    experiment(args)


if __name__ == '__main__':
    seed_everything(76)
    main()
