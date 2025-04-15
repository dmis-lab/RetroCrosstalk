import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import logging
from utils.smiles_process import batch_datas_to_fp, smiles_to_fp, reaction_smarts_to_fp
from utils.prepare_methods import prepare_starting_molecules, prepare_mlp, prepare_egmcts_planner
import arg
import wandb
from api import RSPlanner

def unpack_fps(packed_fps):

    # packed_fps = np.array(packed_fps)
    shape = (*(packed_fps.shape[:-1]), -1)
    # fps = np.unpackbits(packed_fps.reshape((-1, packed_fps.shape[-1])),
    #                    axis=-1)
    fps = torch.FloatTensor(packed_fps).view(shape)

    return fps


class Trainer:
    def __init__(self, model, train_data_loader, n_epochs, lr,
                 save_epoch_int, model_folder, device):
        self.train_data_loader = train_data_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.save_epoch_int = save_epoch_int
        self.model_folder = model_folder
        self.device = device
        self.model = model.to(self.device)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def _pass(self, data, train=True):
        self.optim.zero_grad()
        mol, template, values = data
        fps = unpack_fps(batch_datas_to_fp(mol,template))
        fps = fps.to(self.device)
        values = values.to(self.device)
        v_pred = self.model(fps)
        loss = F.mse_loss(v_pred, values)

        if train:
            loss.backward()
            self.optim.step()

        return loss.item()

    def _train_epoch(self):
        self.model.train()

        losses = []
        pbar = tqdm(self.train_data_loader)
        for data in pbar:
            loss = self._pass(data)
            losses.append(loss)
            pbar.set_description('[loss: %f]' % (loss))

        return np.array(losses).mean()


    def train(self, run):
        best_val_loss = np.inf
        for epoch in range(self.n_epochs):
            self.train_data_loader.reshuffle()

            train_loss = self._train_epoch()
            logging.info(
                '[Epoch %d/%d] [training loss: %f]' %
                (epoch, self.n_epochs, train_loss)
            )
            
            
            run.log({'train_loss': train_loss, 'epoch': epoch})

            if (epoch + 1) % self.save_epoch_int == 0:
                save_file = self.model_folder + '/epoch_%d.pt' % epoch
                torch.save(self.model.state_dict(), save_file, _use_new_zipfile_serialization=False)




            #run.log({'time taken': result['time'], 'route length': result['route_len'],
                     #'iteration': result['iter']})
            
