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


class NetMLP(torch.nn.Module):

    def __init__(self, mat_size):

        super(NetMLP, self).__init__()

        c = 512
        self.fc1 = torch.nn.Linear(mat_size * 15, c)

        d = 6
        self.fc2 = torch.nn.Linear(c, d)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
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

    k = 1.
    w = torch.tensor([1., k, k, k, k, k]).to(device)

    for batch_idx, (data, target, coos, perm) in enumerate(train_loader):

        coos = [c[0].to(device) for c in coos]
        target = target.to(device)
        temp_loss = 0
        #model.module.add_graph(coos, perm)

        for i in range(len(data)):

            output = model(data[i].to(device))
            torch.cuda.synchronize()
            expected = torch.argmax(target[:, i], dim=1)

            loss = F.nll_loss(output, expected, weight=w)
            train_loss += loss
            temp_loss += loss

            for p in model.named_parameters():
                if p[0].split('.')[0][:2] == 'fc':
                    loss = loss + args.reg_weight * (p[1] ** 2).sum()

            loss = loss / mini_batch
            loss.backward()

            if i > 0 and i % mini_batch == 0:
                optimizer.step()
                optimizer.zero_grad()

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

            #model.add_graph(coos, perm)

            for i in range(len(data_t)):
                output = model(data_t[i].to(device))
                torch.cuda.synchronize()
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
            test_loss))

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

    batch_size = 1
    train_loader, test_loader = loaders(device, batch_size=batch_size)

    #train_loader.dataset.self_check()

    data_shape = train_loader.dataset.data_shape()

    model = NetMLP(data_shape)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(1, args.epochs + 1):

        train_loss = train_minibatch(args, model, device, train_loader, optimizer, epoch, mini_batch=45,
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
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    seed_everything(76)
    main()