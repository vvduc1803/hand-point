import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from dex_ycbv import DEX_YCBVDataset
# from timer import Timer
from utils.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config.conf import cfg
from model.network import Model

# dynamic dataset import
# exec('from ' + cfg.trainset + ' import ' + cfg.trainset)
# exec('from ' + cfg.testset + ' import ' + cfg.testset)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # # timer
        # self.tot_timer = Timer()
        # self.gpu_timer = Timer()
        # self.read_timer = Timer()
        #
        # logger
        self.train_logger = colorlogger(cfg.log_dir, log_name=f'train_{log_name}')
        self.test_logger = colorlogger(cfg.log_dir, log_name=f'test{log_name}')

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self, base_path='/home/ana/Study/CVPR/handpoint/dataset'):
        super(Trainer, self).__init__(log_name='logs.txt')
        self.base_path = base_path

    def get_model(self, dic, mode='train'):
        model = Model(dic, mode)
        return model

    def get_optimizer(self, model):
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(model_params, lr=cfg.lr)
        return optimizer

    def save_model(self, state, epoch, best=False, last=False):
        if not best:
            file_path = os.path.join(cfg.model_dir, "snapshot_{}.pth.tar".format(str(epoch)))
        else:
            file_path = os.path.join(cfg.model_dir, "best.pth.tar")

        if last:
            file_path = os.path.join(cfg.model_dir, "last.pth.tar")

        torch.save(state, file_path)
        self.train_logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        # optimizer.load_state_dict(ckpt['optimizer'])

        self.train_logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr * (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr * (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_train_batch_generator(self):
        # data load and construct batch generator
        self.train_logger.info("Creating dataset...")
        self.train_dataset = DEX_YCBVDataset(self.base_path, 'train')

        self.train_itr_per_epoch = math.ceil(len(self.train_dataset) / cfg.num_gpus / cfg.batch_size)
        self.train_batch_generator = DataLoader(dataset=self.train_dataset,
                                                batch_size=cfg.num_gpus * cfg.batch_size,
                                                shuffle=True,
                                                num_workers=cfg.num_thread,
                                                pin_memory=True)

    def _make_test_batch_generator(self):
        # data load and construct batch generator
        self.test_logger.info("Creating test dataset...")
        self.test_dataset = DEX_YCBVDataset(self.base_path, 'test')

        self.test_itr_per_epoch = math.ceil(len(self.test_dataset) / cfg.num_gpus / cfg.batch_size)
        self.test_batch_generator = DataLoader(dataset=self.test_dataset,
                                               batch_size=cfg.num_gpus * cfg.batch_size,
                                               shuffle=False,
                                               num_workers=cfg.num_thread,
                                               pin_memory=True)

    def _make_model(self, mode='train'):
        # prepare network
        self.train_logger.info("Creating graph and optimizer...")
        model = self.get_model(cfg.dic)

        model = model.to('cuda')
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.test_dataset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _eval_result(self):
        return self.test_dataset.eval_result()

    def _mean_eval_result(self):
        return self.test_dataset.mean_eval_result()


class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        self.test_dataset = eval(cfg.testset)(transforms.ToTensor(), "test", cam_scale=cfg.cam_scale)
        self.batch_generator = DataLoader(dataset=self.test_dataset, batch_size=cfg.num_gpus * cfg.batch_size,
                                          shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.test_dataset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, test_epoch):
        return self.test_dataset.print_eval_result(test_epoch)