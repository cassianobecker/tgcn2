from __future__ import print_function
import sys

sys.path.insert(0, '../..')
import argparse
import os
import random
import numpy as np
import scipy.sparse as sp
import torch

import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
from torch.nn import Parameter
from torch_geometric.utils import degree, remove_self_loops
from torch_sparse import spmm

import gcn.graph as graph
from load.data import load_mnist
import math
from datetime import datetime

horizon = 4
in_channels = 3


def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class ChebConvTime(torch.nn.Module):

    def __init__(self, in_channels, out_channels, order_filter, horizon=1, bias=True):
        super(ChebConvTime, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.weight = Parameter(torch.Tensor(K, in_channels, out_channels, h))
        fft_size = int(horizon / 2) + 1
        self.weight = Parameter(torch.Tensor(order_filter, in_channels, out_channels, fft_size, 2))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels * self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        row, col = edge_index
        num_nodes, num_edges, order_filter = x.size(0), row.size(0), self.weight.size(0)

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges,))
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        deg = degree(row, num_nodes, dtype=x.dtype)

        # Compute normalized and rescaled Laplacian.
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        lap = -deg[row] * edge_weight * deg[col]

        def weight_mult(x, w):
            y = torch.einsum('fgrs,ifrs->igrs', w, x)
            return y

        def lap_mult(edge_index, lap, x):
            L = torch.sparse.IntTensor(edge_index, lap, torch.Size([x.shape[0], x.shape[0]])).to_dense()
            x_tilde = torch.einsum('ij,ifrs->jfrs', L, x)
            return x_tilde

        # Perform filter operation recurrently.
        horizon = x.shape[1]
        x = x.permute(0, 2, 1)
        x_hat = torch.rfft(x, 1, normalized=True, onesided=True)

        Tx_0 = x_hat

        y_hat = weight_mult(Tx_0, self.weight[0, :])

        if order_filter > 1:

            Tx_1 = lap_mult(edge_index, lap, x_hat)
            y_hat = y_hat + weight_mult(Tx_1, self.weight[1, :])

            for k in range(2, order_filter):
                Tx_2 = 2 * lap_mult(edge_index, lap, Tx_1) - Tx_0
                y_hat = y_hat + weight_mult(Tx_2, self.weight[k, :])

                Tx_0, Tx_1 = Tx_1, Tx_2

        y = torch.irfft(y_hat, 1, normalized=True, onesided=True, signal_sizes=(horizon,))
        y = y.permute(0, 2, 1)

        if self.bias is not None:
            y = y + self.bias

        return y

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.weight.size(0))


class TimeNet(torch.nn.Module):

    def __init__(self, adj, device, order_filter, out_channels, horizon):
        super(TimeNet, self).__init__()

        num_vertices = adj.shape[0]
        num_classes = 10

        self.edge_index = torch.tensor([adj.tocoo().row, adj.tocoo().col], dtype=torch.long).to(device)

        self.time_conv1 = ChebConvTime(in_channels, out_channels, order_filter=order_filter, horizon=horizon)

        self.fc1 = torch.nn.Linear(num_vertices * out_channels, num_classes)

    def forward(self, x):
        x = self.time_conv1(x, self.edge_index)
        x = F.relu(x)

        # average pool across time-dimension
        x = x.sum(dim=1)

        # fully-connected layer
        x = x.view(-1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=0)


def grid_graph(m, metric='euclidean', number_edges=8, corners=False, shuffled=True):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
    adj = graph.adjacency(dist, idx)

    if shuffled:
        bdj = adj.toarray()
        bdj = list(bdj[np.triu_indices(adj.shape[0])])
        random.shuffle(bdj)
        adj = np.zeros((adj.shape[0], adj.shape[0]))
        indices = np.triu_indices(adj.shape[0])
        adj[indices] = bdj
        adj = adj + adj.T - np.diag(adj.diagonal())
        adj = sp.csr_matrix(adj)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neighbors only.
    if corners:
        adj = adj.toarray()
        adj[adj < adj.max() / 1.5] = 0
        adj = sp.csr_matrix(adj)
        print('{} edges'.format(adj.nnz))

    # print("{} > {} edges".format(adj.nnz // 2, number_edges * m ** 2 // 2))
    return adj


def train(args, model, device, train_loader, optimizer, epoch):
    torch.cuda.synchronize()
    model.train()
    for batch_idx, (data_t, target_t) in enumerate(train_loader):
        data = data_t.to(device)
        target = target_t.to(device)
        optimizer.zero_grad()
        N = data.shape[0]
        outputs = [model(data[i, :].repeat(in_channels, horizon, 1).permute(2, 1, 0)) for i in range(N)]
        targets = [target[i, :].argmax() for i in range(N)]
        loss = sum([F.nll_loss(outputs[i].unsqueeze(0), targets[i].unsqueeze(-1)) for i in range(N)])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {:2d} [{:5d}/{:5d} ({:2.0f}%)] Loss: {:1.5e}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

    torch.cuda.synchronize()


def test(_, model, device, test_loader, epoch):
    model.eval()
    sum_correct = 0.
    test_loss = 0.
    with torch.no_grad():
        for data_t, target_t in test_loader:
            data = data_t.to(device)
            target = target_t.to(device)
            N = data.shape[0]
            outputs = [model(data[i, :].repeat(in_channels, horizon, 1).permute(2, 1, 0)) for i in range(N)]
            preds = [outputs[i].argmax() for i in range(N)]
            targets = [target[i, :].argmax() for i in range(N)]
            correct = sum([targets[i] == preds[i] for i in range(N)]).item()
            sum_correct += correct
            # print(float(correct)/N)
            test_loss += sum([F.nll_loss(outputs[i].unsqueeze(0), targets[i].unsqueeze(-1)) for i in range(N)])

    test_loss /= len(test_loader.dataset)

    # print('')
    print('Epoch: {:3d}, AvgLoss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch, test_loss, float(sum_correct) / len(test_loader.dataset)))
    # print('')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index].astype('float32')
        y = self.labels[index].astype('float32')

        return x, y


def experiment(args):
    if args.output_file:
        fname = 'pytorch_mnist_time_basic' + datetime.now().strftime('%Y%m%d_%H%M%S')
        sys.stdout = open(fname, 'w')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    _, train_data, train_labels, test_data, test_labels = load_mnist()

    train_size = 60000
    test_size = 10000

    train_data = train_data[:train_size, :]
    train_labels = train_labels[:train_size, :]
    test_data = test_data[:test_size, :]
    test_labels = test_labels[:test_size, :]

    training_set = Dataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size)
    validation_set = Dataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size)

    shuffle = False
    order_filter = 15
    out_channels = 10
    number_edges = 8

    print('==========================================')
    print('-- Time Graph Convolution for MNIST ')
    print('edges={:d} | k={:d} | g={:d} | shuffle={:}'
          .format(number_edges, order_filter, out_channels, shuffle))
    print('==========================================')

    adj = grid_graph(28, number_edges=number_edges, corners=False, shuffled=shuffle)
    model = TimeNet(adj, device, order_filter, out_channels, horizon)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(1, args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(args, model, device, test_loader, epoch)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def main():
    # python - u  pygeo_mnist_time_basic.py > out.txt
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--logs-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--output-file', type=bool, default=False, metavar='N',
                        help='logs to file (default=False)')

    args = parser.parse_args()

    experiment(args)


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    seed_everything(1234)

    main()
