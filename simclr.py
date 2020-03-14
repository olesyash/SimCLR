import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import os
import shutil
from data_aug.dataset_wrapper import DataSetWrapper
import numpy as np

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'gpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)

        if n_iter % self.config['log_every_n_steps'] == 0:
            self.writer.add_histogram("xi_repr", ris, global_step=n_iter)
            self.writer.add_histogram("xi_latent", zis, global_step=n_iter)
            self.writer.add_histogram("xj_repr", rjs, global_step=n_iter)
            self.writer.add_histogram("xj_latent", zjs, global_step=n_iter)
            self.writer.add_scalar('train_loss', loss, global_step=n_iter)

        return loss

    def train(self):
        dataset = DataSetWrapper(self.config['batch_size'], **self.config['dataset'])
        train_loader, valid_loader = dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        self._save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                loss.backward()
                optimizer.step()
                n_iter += 1

            if epoch_counter % self.config['eval_every_n_epochs'] == 0:

                # validation steps
                with torch.no_grad():
                    model.eval()

                    valid_loss = 0.0
                    for counter, ((xis, xjs), _) in enumerate(valid_loader):
                        xis = xis.to(self.device)
                        xjs = xjs.to(self.device)

                        loss = self._step(model, xis, xjs, n_iter)
                        valid_loss += loss.item()

                    valid_loss /= counter

                    if valid_loss < best_valid_loss:
                        # save the model weights
                        best_valid_loss = valid_loss
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                    self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                    valid_n_iter += 1

                model.train()

    def _save_config_file(self, model_checkpoints_folder):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model
