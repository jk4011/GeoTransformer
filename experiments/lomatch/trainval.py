import argparse
import time

import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator
import lightning as L
from deepspeed.runtime.lr_schedules import WarmupDecayLR, LRRangeTest


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, lightning=True):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        self.lightning = lightning
        # model, optimizer, scheduler
        if lightning:
            
            self.fabric = L.Fabric(accelerator="cuda", devices=2, strategy="deepspeed_stage_2", precision="16-mixed")
            self.fabric.launch()
            
            
            self.model = create_model(cfg)
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
            
            self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

            self.train_loader = self.fabric.setup_dataloaders(self.train_loader)
            self.val_loader = self.fabric.setup_dataloaders(self.val_loader)

        else:
            model = create_model(cfg).cuda()
            model = self.register_model(model)
            optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
            self.register_optimizer(optimizer)
            scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
            self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)

        loss_dict = self.loss_func(output_dict, data_dict)

        # https://github.com/pytorch/pytorch/issues/43259#issuecomment-964284292
        for param in self.model.parameters():
            loss_dict['loss'] += param.sum() * 0

        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
