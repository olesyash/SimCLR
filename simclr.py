import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint, acc

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def test(self, test_loader, test_set_len, device):
        # save config file
        save_config_file(self.writer.log_dir, self.args)
        for param in self.model.parameters():
            param.requires_grad = False

        top1_accuracy = 0
        logging.info(f"Start SimCLR test")
        logging.info(f"Testing with gpu: {not self.args.disable_cuda}.")

        for counter, (images, labels) in enumerate(test_loader):
            images = torch.cat(images, dim=0)
            images = images.to(device)
            labels = labels.to(device)
        
            logits = self.model(images)
        
            top1, top5 = accuracy(logits, labels, topk=(1,5))
            top1_accuracy += top1[0]
        top1_accuracy /= (counter + 1)

        logging.info(f"Test accuracy: {top1_accuracy.item()}")
        logging.info("Test has finished.")
    
    # def test(self, test_loader, test_set_len):
    #     # save config file
    #     save_config_file(self.writer.log_dir, self.args)

    #     running_corrects = 0
    #     logging.info(f"Start SimCLR test")
    #     logging.info(f"Testing with gpu: {not self.args.disable_cuda}.")

    #     for images, labels in tqdm(test_loader):
    #         images = torch.cat(images, dim=0)
    #         images = images.to(self.args.device)
    #         labels = labels.to(self.args.device)
    #         for param in self.model.parameters():
    #             param.requires_grad = False

    #         with autocast(enabled=self.args.fp16_precision):
    #             features = self.model(images)
    #             logits, _ = self.info_nce_loss(features)
    #             _, preds = torch.max(logits, 1)
    #             print("Size of logits: ", logits.size())
    #             print("Size of preds: ", preds.size())
    #             print("Size of labels: ", labels.size())
    #             running_corrects += torch.sum(preds == labels)

    #     logging.info(f"Running_corrects: {running_corrects}")
    #     epoch_acc = running_corrects.double() / test_set_len * 100    
    #     logging.info(f"Test accuracy: {epoch_acc}")
    #     logging.info("Test has finished.")

