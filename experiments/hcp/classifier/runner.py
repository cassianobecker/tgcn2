import os
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from util.logging import set_logger, get_logger
from util.experiment import print_memory


class Runner:

    def __init__(self, device, params, train_loader, test_loader):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        log_furl = os.path.join(params['FILE']['experiment_path'], 'log', 'experiment.log')
        set_logger('Experiment', params['LOGGING']['experiment_level'], log_furl)
        self.experiment_logger = get_logger('Experiment')

        log_furl = os.path.join(params['FILE']['experiment_path'], 'log', 'monitor.log')
        set_logger('Monitor', params['LOGGING']['monitor_level'], log_furl)
        self.monitor_logger = get_logger('Monitor')

        self.monitor_logger.info('creating runner class')
        # self.monitor_logger.info({section: dict(params[section]) for section in params.sections()})

    def run(self, args, model):

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model.to(self.device)

        self.monitor_logger.info('starting experiment')

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
        :return: train_loss
        """
        train_loss = 0
        model.train()

        k = 1.
        w = torch.tensor([1., k, k, k, k, k]).to(self.device)

        for batch_idx, (bold_ts, targets, graph_list, mapping_list, subject) in enumerate(self.train_loader):

            if subject is None:
                self.monitor_logger.warning('Empty data for subject {:}, skipping', subject[0])
                continue

            self.monitor_logger.info('training on subject {:}'.format(subject[0]))
            self.monitor_logger.info(print_memory())

            targets = targets.to(self.device)
            graph_list = [c[0].to(self.device) for c in graph_list]

            temp_loss = 0

            for i in range(len(bold_ts)):

                self.monitor_logger.debug('before output: ' + print_memory())
                output = model(bold_ts[i].to(self.device), graph_list, mapping_list)
                self.monitor_logger.debug('after output: ' + print_memory())

                expected = torch.argmax(targets[:, i], dim=1)

                torch.cuda.synchronize()
                loss = F.nll_loss(output, expected, weight=w)

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

        train_loss /= (len(self.train_loader.dataset) * len(bold_ts))

        return train_loss

    def test_batch(self, args, model, epoch, verbose=True):
        """
        Evaluates the model trained in train_minibatch() on patients loaded from the test set.
        :return: test_loss and correct, the # of correct predictions
        """
        model.eval()

        test_loss = 0
        correct = 0

        preds = torch.empty(0, dtype=torch.long).to(self.device)
        targets = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.no_grad():

            for batch_idx, (data_t, target_t, coos, perm, subject) in enumerate(self.test_loader):

                if subject is None:
                    self.monitor_logger.warning('dmpty test batch number {:}, skipping', batch_idx)
                    continue

                self.monitor_logger.info('testing on subject {:}'.format(subject[0]))

                target = target_t.to(self.device)
                coos = [c[0].to(self.device) for c in coos]

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
