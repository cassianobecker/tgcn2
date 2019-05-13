import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F


class Runner:

    def __init__(self, device, train_loader, test_loader):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def run(self, args, model):

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model.to(self.device)

        for epoch in range(1, args.epochs + 1):
            train_loss = self.train_minibatch(args, model, epoch, optimizer, mini_batch=45, verbose=True)
            scheduler.step()

            test_loss, correct = self.test_batch(args, model, epoch)

            print('Epoch: {} Training loss: {:1.3e}, Test loss: {:1.3e}, Accuracy: {}/{} ({:.2f}%)'.format(
                epoch, train_loss, test_loss, correct,
                len(self.test_loader.dataset) * 270,
                100. * correct / (len(self.test_loader.dataset) * 270)
            ))

        if args.save_model:
            torch.save(model.state_dict(), "hcp_cnn_1gpu2.pt")

    def train_minibatch(self, args, model, epoch, optimizer, mini_batch=10, verbose=True):
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
        w = torch.tensor([1., k, k, k, k, k]).to(self.device)

        for batch_idx, (data, target, graph_list, perm) in enumerate(self.train_loader):

            target = target.to(self.device)

            graph_list = [c[0].to(self.device) for c in graph_list]

            # if torch.cuda.device_count() > 1:
            #     model.module.set_graph(graph_list, perm)
            # else:
            #     model.set_graph(graph_list, perm)

            temp_loss = 0

            for i in range(len(data)):

                output = model(data[i].to(self.device), graph_list, perm)

                torch.cuda.synchronize()
                expected = torch.argmax(target[:, i], dim=1)

                loss = F.nll_loss(output, expected, weight=w)

                # for p in model.named_parameters():
                #     if p[0].split('.')[0][:2] == 'fc':
                #         loss = loss + args.reg_weight * (p[1] ** 2).sum()

                train_loss += loss.item()
                temp_loss += loss.item()

                loss = loss / mini_batch
                loss.backward()

                if i > 0 and i % mini_batch == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            if verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx + 1, len(self.train_loader.dataset),
                           100. * (batch_idx + 1) / len(self.train_loader.dataset),
                    temp_loss))

        train_loss /= (len(self.train_loader.dataset) * len(data))

        return train_loss

    def test_batch(self, args, model, epoch, verbose=True):
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

        preds = torch.empty(0, dtype=torch.long).to(self.device)
        targets = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.no_grad():
            for batch_idx, (data_t, target_t, coos, perm) in enumerate(self.test_loader):

                target = target_t.to(self.device)

                coos = [c[0].to(self.device) for c in coos]

                # if torch.cuda.device_count() > 1:
                #     model.module.set_graph(coos, perm)
                # else:
                #     model.set_graph(coos, perm)

                for i in range(len(data_t)):
                    output = model(data_t[i].to(self.device), coos, perm)

                    torch.cuda.synchronize()

                    expected = torch.argmax(target[:, i], dim=1)
                    test_loss += F.nll_loss(output, expected, reduction='sum').item()

                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                    preds = torch.cat((pred, preds))
                    targets = torch.cat((expected, targets))
                    correct += pred.eq(expected.view_as(pred)).sum().item()

        test_loss /= (len(self.test_loader.dataset) * len(data_t))

        if verbose:
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx + 1,
                len(self.test_loader.dataset),
                       100. * (batch_idx + 1) / len(self.test_loader.dataset),
                test_loss
            ))

        return test_loss, correct
