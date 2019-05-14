import os
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, accuracy_score
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

        self.name = params['FILE']['experiment_name']
        self.path = params['FILE']['experiment_path']

        k = 1.
        self.w = torch.tensor([1., k, k, k, k, k]).to(self.device)

    def run(self, args, model):

        mini_batch = 10

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(self.device)

        self.monitor_logger.info('starting experiment')

        for epoch in range(1, args.epochs + 1):

            train_loss_value, predictions, targets = self.train_minibatch(model, epoch, optimizer, mini_batch)

            self.print_eval(train_loss_value, predictions, targets, idx=epoch, header='Train epoch:')
            self.print_confusion_matrix(predictions, targets)

            scheduler.step()

            test_loss_value, predictions, targets = self.test_batch(model, epoch)

            self.print_eval(test_loss_value, predictions, targets, idx=epoch, header='Test epoch:')
            self.print_confusion_matrix(predictions, targets)

            if args.save_model:
                model_furl = os.path.join(self.path, 'out', 'model_epoch_{:}'.format(epoch) + '.pt')
                torch.save(model.state_dict(), model_furl)

    def train_minibatch(self, model, epoch, optimizer, mini_batch=10):
        """
        Loads input data (BOLD signal windows and corresponding target motor tasks) from one patient at a time,
        and minibatches the windowed input signal while training the TGCN by optimizing for minimal training loss.
        :return: train_loss
        """
        model.train()

        train_loss_value = 0
        predictions = []
        targets = []

        for batch_idx, (bold_ts, cues, graph_list, mapping_list, subject) in enumerate(self.train_loader):

            if subject is None:
                self.monitor_logger.warning('empty training batch, skipping')
                continue

            self.monitor_logger.info('training on subject {:} ({:} of {:})'.
                                     format(subject[0], batch_idx + 1, len(self.train_loader.dataset.subjects)))

            self.monitor_logger.info(print_memory())

            cues = cues.to(self.device)
            graph_list = [g[0].to(self.device) for g in graph_list]

            batch_loss_value = 0
            batch_predictions = []
            batch_targets = []

            for i in range(len(bold_ts)):

                output = model(bold_ts[i].to(self.device), graph_list, mapping_list)
                target = torch.argmax(cues[:, i], dim=1)
                prediction = output.max(1, keepdim=True)[1][0]

                torch.cuda.synchronize()

                loss = F.nll_loss(output, target, weight=self.w)
                loss = loss / mini_batch
                loss.backward()

                if i > 0 and i % mini_batch == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss_value += loss.item()
                predictions.extend(prediction.tolist())
                targets.extend(target.tolist())

                batch_loss_value += loss.item()
                batch_predictions.extend(prediction.tolist())
                batch_targets.extend(target.tolist())

            self.print_eval(batch_loss_value, batch_predictions, batch_targets, idx=batch_idx, header='batch idx:')

        return train_loss_value, predictions, targets

    def test_batch(self, model, epoch):
        """
        Evaluates the model trained in train_minibatch() on patients loaded from the test set.
        :return: test_loss and correct, the # of correct predictions
        """
        model.eval()

        test_loss_value = 0
        predictions = []
        targets = []

        with torch.no_grad():

            for batch_idx, (bold_ts, cues, graph_list, mapping_list, subject) in enumerate(self.test_loader):

                if subject is None:
                    self.monitor_logger.warning('empty test batch, skipping')
                    continue

                self.monitor_logger.info('testing on subject {:} ({:} of {:})'.
                                         format(subject[0], batch_idx + 1, len(self.test_loader.dataset.subjects)))

                cues = cues.to(self.device)
                graph_list = [g[0].to(self.device) for g in graph_list]

                for i in range(len(bold_ts)):
                    output = model(bold_ts[i].to(self.device), graph_list, mapping_list)
                    target = torch.argmax(cues[:, i], dim=1)
                    prediction = output.max(1, keepdim=True)[1][0]

                    torch.cuda.synchronize()

                    test_loss_value += F.nll_loss(output, target, reduction='sum').item()

                    predictions.extend(prediction.tolist())
                    targets.extend(target.tolist())

        return test_loss_value, predictions, targets

    def print_eval(self, loss_value, predictions, targets, idx=None, header=''):

        accuracy = accuracy_score(targets, predictions)

        msg = f'| {header:14s}' + f'{idx:3d} | '
        msg = msg + f'loss: {loss_value:1.3e} | '
        msg = msg + f'accuracy: {accuracy: 1.3f} |'

        self.experiment_logger.info(msg)
        print(msg)

    def print_confusion_matrix(self, predictions, targets):

        cm = confusion_matrix(targets, predictions)
        self.experiment_logger.info(cm)
        print(cm)
