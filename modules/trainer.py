import logging
import os
import shutil
import time
from abc import abstractmethod

import torch
import wandb
from numpy import inf


def load_checkpoint(path, map_location='cpu'):
    # Checkpoints hold numpy scalars (monitor_best, best_recorder), so the
    # weights_only=True default of PyTorch >= 2.6 cannot unpickle them.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 1.13 has no weights_only argument
        return torch.load(path, map_location=map_location)


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        # Test metrics are calculated once after training, not used for checkpoint selection.
        # self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.not_improved_count = 0

        # Graceful stop before the platform kills the session (e.g. Kaggle's 12h limit).
        self.time_limit = getattr(args, 'time_limit_hours', 0) * 3600
        self.session_start = getattr(args, 'session_start', time.time())
        self.stopped_by_time = False

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}
        # The previous per-epoch test recorder is intentionally disabled:
        # 'test': {self.mnt_metric_test: self.mnt_best}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        if self.start_epoch > self.epochs:
            self.logger.warning(
                "start_epoch ({}) > --epochs ({}): no training epochs will run. "
                "--epochs is the total target, raise it to continue training.".format(self.start_epoch, self.epochs))
        last_epoch = self.start_epoch - 1
        train_start = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            wandb.log({
                'epoch': epoch,
                'train_loss': log['train_loss'],
                'val_loss': log['val_loss'],
                'val_BLEU_1': log['val_BLEU_1'],
                'val_BLEU_2': log['val_BLEU_2'],
                'val_BLEU_3': log['val_BLEU_3'],
                'val_BLEU_4': log['val_BLEU_4'],
            }, step=epoch)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                    best = True
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            saved = False
            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best)
                saved = True

            if epoch < self.epochs and self._time_limit_reached(epoch, train_start):
                if not saved:
                    self._save_checkpoint(epoch)
                self.stopped_by_time = True
                self.logger.info(
                    "Time limit of {:.1f}h reached after epoch {}. Training stops; resume with "
                    "--resume {}".format(self.time_limit / 3600, epoch,
                                         os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')))
                break

        if self.stopped_by_time:
            self.logger.info("Stopped by time limit: skipping test evaluation until training finishes.")
            return

        self._evaluate_best_on_test(step=last_epoch + 1)

    def _time_limit_reached(self, epoch, train_start):
        if self.time_limit <= 0:
            return False
        now = time.time()
        elapsed = now - self.session_start
        avg_epoch = (now - train_start) / (epoch - self.start_epoch + 1)
        # stop if the next epoch (with a 20% margin) would cross the limit
        projected = elapsed + 1.2 * avg_epoch
        self.logger.info("Elapsed {:.2f}h of {:.2f}h limit, avg epoch {:.1f} min.".format(
            elapsed / 3600, self.time_limit / 3600, avg_epoch / 60))
        return projected > self.time_limit

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        # Test-set best tracking used to happen every epoch. It is disabled so test
        # metrics remain independent until the selected validation-best checkpoint is evaluated.
        # improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
        #     self.mnt_metric_test]) or \
        #                 (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
        #                     self.mnt_metric_test])
        # if improved_test:
        #     self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        # Test metrics are reported once by _evaluate_best_on_test(), not as an epoch-best.
        # self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        # for key, value in self.best_recorder['test'].items():
        #     self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    @staticmethod
    def _atomic_save(state, path):
        # write to a temp file first so a kill mid-write never corrupts the previous checkpoint
        tmp_path = path + '.tmp'
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'monitor_best': self.mnt_best,
            'not_improved_count': self.not_improved_count,
            'best_recorder': self.best_recorder,
            'wandb_run_id': wandb.run.id if wandb.run is not None else None,
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        self._atomic_save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            self._atomic_save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = load_checkpoint(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'lr_scheduler' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            # older checkpoint: the current lr is already restored in the optimizer state,
            # only the scheduler's epoch counter needs to catch up (step() runs once per epoch)
            self.lr_scheduler.last_epoch = checkpoint['epoch']
            self.logger.warning("Checkpoint has no lr_scheduler state; set last_epoch={}.".format(checkpoint['epoch']))

        self.not_improved_count = checkpoint.get('not_improved_count', 0)
        self.best_recorder = checkpoint.get('best_recorder', {'val': {self.mnt_metric: self.mnt_best}})

        # _evaluate_best_on_test() reads model_best.pth from save_dir; bring it along
        # when resuming from another (e.g. read-only Kaggle input) directory.
        best_src = os.path.join(os.path.dirname(resume_path), 'model_best.pth')
        best_dst = os.path.join(self.checkpoint_dir, 'model_best.pth')
        if os.path.isfile(best_src) and not os.path.isfile(best_dst):
            shutil.copyfile(best_src, best_dst)
            self.logger.info("Copied {} to {}".format(best_src, best_dst))

        self.logger.info("Checkpoint loaded. Resume training from epoch {} (not_improved_count={}, lr={})".format(
            self.start_epoch, self.not_improved_count, [g['lr'] for g in self.optimizer.param_groups]))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):

            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                 reports_masks.to(self.device)
            output = self.model(images, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1)))

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            val_loss = 0.0
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                val_output = self.model(images, reports_ids, mode='train')
                loss = self.criterion(val_output, reports_ids, reports_masks)
                val_loss += loss.item()

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            log['val_loss'] = val_loss / len(self.val_dataloader)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        # Per-epoch test evaluation was moved to _evaluate_best_on_test().
        # self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        # self.model.eval()
        # with torch.no_grad():
        #     test_gts, test_res = [], []
        #     for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
        #         images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
        #             self.device), reports_masks.to(self.device)
        #         output, _ = self.model(images, mode='sample')
        #         reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
        #         ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        #         test_res.extend(reports)
        #         test_gts.extend(ground_truths)
        #
        #     test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
        #                                 {i: [re] for i, re in enumerate(test_res)})
        #     log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log

    def _evaluate_best_on_test(self, step):
        best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        if not os.path.isfile(best_path):
            self.logger.warning(
                "Best checkpoint '{}' was not found; skipping test evaluation.".format(best_path))
            return

        self.logger.info('Evaluating test set once using best checkpoint: {}'.format(best_path))
        checkpoint = load_checkpoint(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        test_gts, test_res = [], []
        with torch.no_grad():
            for images_id, images, reports_ids, reports_masks in self.test_dataloader:
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

        test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
        test_log = {'test_' + key: value for key, value in test_met.items()}

        for key, value in test_log.items():
            self.logger.info('\t{:15s}: {}'.format(key, value))

        wandb.log(test_log, step=step)
