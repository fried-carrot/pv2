#!/usr/bin/env python3
"""
GMVAE training
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import os
import json
import argparse
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio


class GMVAE_ZINB(nn.Module):
    # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py
    # og: class GMVAE_ZINB(nn.Module):

    def __init__(self, args):
        super(GMVAE_ZINB, self).__init__()
        self.device = args.device
        self.args = args
        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py line 11-12
        # og: self.onehot = torch.nn.functional.one_hot(torch.arange(0, self.args.K), num_classes=self.args.K).to(self.device)*(1.0)
        self.onehot = torch.nn.functional.one_hot(torch.arange(0, self.args.K),
                                            num_classes=self.args.K).to(self.device)*(1.0)

        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 14-15
        # og: self.pi1 = nn.Linear(self.args.input_dim, self.args.h_dim)
        # og: self.pi2 = nn.Linear(self.args.h_dim, self.args.K)
        self.pi1 = nn.Linear(self.args.input_dim, self.args.h_dim)
        self.pi2 = nn.Linear(self.args.h_dim, self.args.K)

        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 17-23
        # og: encoder mu path
        self.mu_x1 = nn.Linear(self.args.input_dim + self.args.K, self.args.h_dim1)
        self.mu_x2 = nn.Linear(self.args.h_dim1, self.args.h_dim2)
        self.mu_x3 = nn.Linear(self.args.h_dim2, self.args.z_dim)
        # og: encoder logvar path
        self.logvar_x1 = nn.Linear(self.args.input_dim + self.args.K, self.args.h_dim1)
        self.logvar_x2 = nn.Linear(self.args.h_dim1, self.args.h_dim2)
        self.logvar_x3 = nn.Linear(self.args.h_dim2, self.args.z_dim)

        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 25-29
        # og: mixture component parameters
        self.mu_w1 = nn.Linear(self.args.K, self.args.h_dim)
        self.mu_w2 = nn.Linear(self.args.h_dim, self.args.z_dim)
        self.logvar_w1 = nn.Linear(self.args.K, self.args.h_dim)
        self.logvar_w2 = nn.Linear(self.args.h_dim, self.args.z_dim)

        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 31-40
        # og: decoder layers
        self.recon1 = nn.Linear(self.args.z_dim, self.args.h_dim2)
        self.recon2z = nn.Linear(self.args.h_dim2, self.args.h_dim1)
        self.recon3z = nn.Linear(self.args.h_dim1, self.args.input_dim)
        self.recon2m = nn.Linear(self.args.h_dim2, self.args.h_dim1)
        self.recon3m = nn.Linear(self.args.h_dim1, self.args.input_dim)
        self.recon2d = nn.Linear(self.args.h_dim2, self.args.h_dim1)
        self.recon3d = nn.Linear(self.args.h_dim1, self.args.input_dim)

        # regularization layers (inspired by Ioffe & Szegedy 2015, Srivastava et al. 2014)
        self.bn_h_dim1 = nn.BatchNorm1d(self.args.h_dim1)
        self.bn_h_dim2 = nn.BatchNorm1d(self.args.h_dim2)
        self.dropout = nn.Dropout(0.1)

    def pi_of_x(self,x):
        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 42-45
        # og: x = F.relu(self.pi1(x))
        # og: pi_x = F.softmax(self.pi2(x), dim=1)
        x = F.relu(self.pi1(x))
        pi_x = F.softmax(self.pi2(x), dim=1)
        return pi_x

    def musig_of_z(self,x):
        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 47-60
        batchSize = x.size(0)
        mu_zs = torch.empty(batchSize, self.args.z_dim, self.args.K,
                      device=self.device, requires_grad=False)
        logvar_zs = torch.empty(batchSize, self.args.z_dim, self.args.K,
                          device=self.device, requires_grad=False)
        for i in range(self.args.K):
            xy = torch.cat((x, self.onehot[i,:].expand(x.size(0),self.args.K)), 1)
            # apply batch norm and dropout for regularization
            h1_mu = self.dropout(F.relu(self.bn_h_dim1(self.mu_x1(xy))))
            h2_mu = self.dropout(F.relu(self.bn_h_dim2(self.mu_x2(h1_mu))))
            mu_zs[:, :, i] = self.mu_x3(h2_mu)

            h1_logvar = self.dropout(F.relu(self.bn_h_dim1(self.logvar_x1(xy))))
            h2_logvar = self.dropout(F.relu(self.bn_h_dim2(self.logvar_x2(h1_logvar))))
            logvar_zs[:, :, i] = self.logvar_x3(h2_logvar)
        return mu_zs, logvar_zs

    def musig_of_genz(self,y,batchsize):
        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 62-66
        mu_genz = self.mu_w2(F.relu(self.mu_w1(y)))
        logvar_genz = self.logvar_w2(F.relu(self.logvar_w1(y)))
        return torch.t(mu_genz).expand(batchsize, self.args.z_dim, self.args.K),\
            torch.t(logvar_genz).expand(batchsize, self.args.z_dim, self.args.K)

    def decoder(self, z_sample):
        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 69-89
        h = F.relu(self.recon1(z_sample))

        hz = F.relu(self.recon2z(h))  #zero_probability
        hz = self.recon3z(hz)         #zero_probability

        hm = F.relu(self.recon2m(h))  #mean
        hm = self.recon3m(hm)         #mean

        hd = F.relu(self.recon2d(h))  #dispersion
        hd = self.recon3d(hd)         #dispersion

        Yzerop = F.sigmoid(hz)   #scRNA zinb zero probability
        # og: Ymean = torch.exp(hm)   #scRNA zinb mean    # This cause nan
        # og: Ydisper = torch.exp(hd)   #scRNA zinb dispersion  # This cause nan
        # og: Ymean = F.relu(hm)+1e-10   #scRNA zinb mean   # This didn't cause nan
        # og: Ydisper = F.relu(hd)+1e-10   #scRNA zinb dispersion  # This didn't cause nan
        Ymean = nn.ELU(1)(hm*(1.0))+1.0
        Ydisper = nn.ELU(1)(hd*(1.0))+1.0
        return Yzerop, Ymean, Ydisper

    def reparameterize(self, mu, logvar):
        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 91-104
        if self.training:
            # do this only while training
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, Xdata, Xtarget):
        # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_scrna_zinb.py lines 106-132
        batchsize = Xdata.size(0)
        Xdata=Xdata.to(self.device)
        Xtarget=Xtarget.to(self.device)

        X_conv = Xdata
        pi_x = self.pi_of_x(X_conv)
        mu_zs, logvar_zs = self.musig_of_z(X_conv)

        mu_genz, logvar_genz = self.musig_of_genz(self.onehot, batchsize)
        z_samples = self.reparameterize(mu_zs, logvar_zs)
        x_recons_zerop = torch.stack([self.decoder(z_samples[:,:,ii].squeeze())[0] \
                                                for ii in (range(z_samples.size(2)))], dim=len(z_samples[:,:,0].squeeze().shape)\
                                                    )
        x_recons_mean = torch.stack([self.decoder(z_samples[:,:,ii].squeeze())[1] \
                                                for ii in (range(z_samples.size(2)))], dim=len(z_samples[:,:,0].squeeze().shape)\
                                                    )
        x_recons_disper = torch.stack([self.decoder(z_samples[:,:,ii].squeeze())[2] \
                                                for ii in (range(z_samples.size(2)))], dim=len(z_samples[:,:,0].squeeze().shape)\
                                                    )
        pi_x_expanded = pi_x.unsqueeze(1).expand(batchsize,X_conv.size(-1),self.args.K)
        x_recon_zerop = torch.sum(x_recons_zerop*pi_x_expanded, dim=pi_x_expanded.dim()-1)
        x_recon_mean = torch.sum(x_recons_mean*pi_x_expanded, dim=pi_x_expanded.dim()-1)
        x_recon_disper = torch.sum(x_recons_disper*pi_x_expanded, dim=pi_x_expanded.dim()-1)
        return pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz, x_recons_zerop, x_recons_mean, \
            x_recons_disper, x_recon_zerop, x_recon_mean, x_recon_disper

    def freeze_model(self):
        # for downstream  integration
        for param in self.parameters():
            param.requires_grad = False
        print("GMVAE model frozen")

    def get_embeddings(self, x):
        # for P4P patient classification
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            # use mixture assignment probabilities to weight embeddings
            pi_x = self.pi_of_x(x)
            mu_zs, logvar_zs = self.musig_of_z(x)
            # return weighted sum across mixture components
            # pi_x: (batch, K), mu_zs: (batch, z_dim, K)
            # expand pi_x to (batch, z_dim, K) and sum over K
            weighted_mu = (mu_zs * pi_x.unsqueeze(1)).sum(dim=2)  # (batch, z_dim)
            return weighted_mu


def gmvae_losses(Xdata, Xtarget, pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz,\
    x_recons_zerop, x_recons_mean, x_recons_disper, epoch=0):
    # from: bulk2sc_GMVAE/_model_source_codes/GMVAE_losses_zinb.py lines 5-67
    # adaptive loss weighting inspired by Lopez et al. 2018 (scVI) and Kendall et al. 2018
    eps = 1e-10
    batchsize = Xdata.size(0)
    K = pi_x.size(1)
    z_dim = mu_zs.size(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xdata = Xdata.to(device)
    pi_target = torch.nn.functional.one_hot(Xtarget, num_classes=K).to(device)

    KLD_pi = torch.sum(pi_x*(torch.log(pi_x + eps)-torch.log(pi_target + eps)))

    KLD_gaussian = 0.5 * (((logvar_genz - logvar_zs) + \
        ((logvar_zs.exp() + (mu_zs - mu_genz).pow(2))/logvar_genz.exp())) - 1)

    KLD_gaussian = torch.sum(KLD_gaussian * pi_x.unsqueeze(1).expand(batchsize,z_dim,K))

    pi_x_expanded = pi_x.unsqueeze(1).expand(batchsize, Xdata.shape[-1], K)

    # zinb_loss_i
    # zero case
    zero_cases_i = torch.stack([((-1.0)*torch.log( x_recons_zerop[:,:,ii] \
        + (1 - x_recons_zerop[:,:,ii])\
            *torch.pow(((x_recons_disper[:,:,ii])/(x_recons_mean[:,:,ii] + x_recons_disper[:,:,ii] + eps)),\
                x_recons_disper[:,:,ii]) + eps)) for ii in range(K)], dim=x_recons_disper.dim()-1)

    # non zero case
    nzero_cases1_i = torch.stack([(torch.lgamma(Xdata+1) + torch.lgamma(x_recons_disper[:,:,ii] + eps) \
        + (-1.0)*torch.lgamma(Xdata + x_recons_disper[:,:,ii]+eps)  ) for ii in range(K)], \
            dim=x_recons_disper.dim()-1)

    nzero_cases2_i = torch.stack([( (-1.0)*torch.log(1.0 - x_recons_zerop[:,:,ii] + eps) \
     + (-1.0)*x_recons_disper[:,:,ii]*torch.log(x_recons_disper[:,:,ii] + eps) \
         + (-1.0)*Xdata*torch.log(x_recons_mean[:,:,ii] + eps) \
             + (x_recons_disper[:,:,ii] + Xdata)*torch.log(x_recons_disper[:,:,ii] + x_recons_mean[:,:,ii] + eps) ) \
                 for ii in range(K)], dim=x_recons_disper.dim()-1)

    nzero_cases_i = nzero_cases1_i + nzero_cases2_i

    #Choose one case
    Xdata_expanded = Xdata.unsqueeze(Xdata.dim()).expand(batchsize,Xdata.size(1),K)

    zinb_losses = torch.where(torch.le(Xdata_expanded, 0.01), zero_cases_i, nzero_cases_i)

    #Sum them up
    zinb_loss = torch.sum(zinb_losses*pi_x_expanded)

    # adaptive weighting with warmup schedule
    # starts at 1.0, increases to 30.0 over training to prioritize cell-type learning
    alpha = min(1.0 + epoch * 0.3, 30.0)
    total_loss = KLD_gaussian + alpha * KLD_pi + zinb_loss

    return total_loss, KLD_gaussian, KLD_pi, zinb_loss


def contrastive_loss(z, cell_types, temperature=0.5):
    """
    contrastive loss for self-supervised learning
    inspired by Yang et al. 2022 (CLEAR) and Chen et al. 2020 (SimCLR)
    pulls same cell-type embeddings together, pushes different types apart

    args:
        z: (batch, z_dim) - GMVAE embeddings
        cell_types: (batch,) - cell type labels
        temperature: temperature parameter for scaling similarities
    """
    # normalize embeddings
    z_norm = F.normalize(z, dim=1)

    # compute pairwise similarities
    sim_matrix = torch.matmul(z_norm, z_norm.t()) / temperature

    # create mask for positive pairs (same cell type)
    labels = cell_types.unsqueeze(1)
    mask_positive = (labels == labels.t()).float()
    mask_positive.fill_diagonal_(0)  # exclude self-pairs

    # avoid numerical issues
    exp_sim = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach())

    # log probability for positive pairs
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-10)

    # mean over positive pairs per sample
    pos_per_sample = mask_positive.sum(dim=1)
    pos_per_sample = torch.clamp(pos_per_sample, min=1.0)  # avoid division by zero

    loss = -(mask_positive * log_prob).sum(dim=1) / pos_per_sample

    return loss.mean()


# create args object for GMVAE (new)
class Args:
    def __init__(self, input_dim, n_cell_types, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.K = n_cell_types  # K mixture components = number of cell types
        self.z_dim = 64
        self.h_dim = 512
        self.h_dim1 = 512
        self.h_dim2 = 256
        self.device = device


def train_gmvae(data_loader, input_dim, n_cell_types, save_path, epochs=100,
                learning_rate=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):

    # create args for original bulk2sc model (new)
    args = Args(input_dim, n_cell_types, device)

    model = GMVAE_ZINB(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # learning rate scheduler with warmup and cosine annealing (Loshchilov & Hutter 2017)
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # cosine annealing
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"training GMVAE on {device}")
    print(f"input dimension: {input_dim}")
    print(f"cell types: {n_cell_types}")
    print(f"mixture components (K): {args.K}")
    print(f"learning rate schedule: warmup {warmup_epochs} epochs, then cosine annealing")
    print()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        n_batches = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (batch, labels) in enumerate(data_loader):
            x = batch.to(device)
            # use cell type labels
            targets = labels.to(device)

            optimizer.zero_grad()

            # forward pass
            pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz, x_recons_zerop, x_recons_mean, \
                x_recons_disper, x_recon_zerop, x_recon_mean, x_recon_disper = model(x, targets)

            # losses (pass epoch for adaptive weighting)
            total_loss_batch, KLD_gaussian, KLD_pi, zinb_loss = gmvae_losses(
                x, targets, pi_x, mu_zs, logvar_zs, z_samples, mu_genz, logvar_genz,
                x_recons_zerop, x_recons_mean, x_recons_disper, epoch=epoch)

            # add contrastive regularization (inspired by Yang et al. 2022)
            z_embeddings = model.get_embeddings(x)
            contrast_loss = contrastive_loss(z_embeddings, targets)

            # total loss with contrastive term (small weight for regularization)
            total_loss_batch = total_loss_batch + 0.1 * contrast_loss

            # backward pass
            total_loss_batch.backward()

            # gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += total_loss_batch.item()
            total_recon_loss += zinb_loss.item()
            total_kld_loss += (KLD_gaussian + KLD_pi).item()
            n_batches += 1

            # cell type classification accuracy
            predicted_cell_types = pi_x.argmax(dim=1)
            correct_predictions += (predicted_cell_types == targets).sum().item()
            total_predictions += targets.size(0)

        avg_loss = total_loss / n_batches
        avg_recon = total_recon_loss / n_batches
        avg_kld = total_kld_loss / n_batches
        cell_type_acc = correct_predictions / total_predictions

        # update learning rate
        scheduler.step()

        if epoch % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"epoch {epoch:3d}: loss={avg_loss:.4f}, recon={avg_recon:.4f}, kld={avg_kld:.4f}, cell_type_acc={cell_type_acc:.4f}, lr={current_lr:.6f}")

    # compute and store global cell type priors (for z-score calculation in classifier)
    model.eval()
    with torch.no_grad():
        # compute global means and logvars for each mixture component
        mu_genz, logvar_genz = model.musig_of_genz(model.onehot, batchsize=1)
        # mu_genz, logvar_genz: (1, z_dim, K) -> transpose to (K, z_dim)
        model.mu_prior = mu_genz.squeeze(0).transpose(0, 1)  # (K, z_dim)
        model.logvar_prior = logvar_genz.squeeze(0).transpose(0, 1)  # (K, z_dim)
        print(f"stored global priors: mu_prior {model.mu_prior.shape}, logvar_prior {model.logvar_prior.shape}")

    # freeze model
    model.freeze_model()

    # save trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'mu_prior': model.mu_prior,
        'logvar_prior': model.logvar_prior,
        'training_config': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'final_loss': avg_loss,
        }
    }, save_path)

    print(f"GMVAE saved to: {save_path}")
    return model


def load_frozen_gmvae(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # for P4P patient classification
    checkpoint = torch.load(model_path, map_location=device)

    model = GMVAE_ZINB(checkpoint['args']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # restore global priors
    model.mu_prior = checkpoint['mu_prior'].to(device)
    model.logvar_prior = checkpoint['logvar_prior'].to(device)

    model.freeze_model()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train GMVAE-4P')
    parser.add_argument('--data_dir', required=True, help='preprocessed data directory')
    parser.add_argument('--output', required=True, help='output model path')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')

    args = parser.parse_args()

    print("=" * 60)
    print("GMVAE training")
    print("=" * 60)

    # load metadata
    with open(os.path.join(args.data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    n_genes = metadata['n_genes']
    n_cell_types = metadata['n_cell_types']
    print(f"dataset: {metadata['n_cells']} cells x {n_genes} genes")
    print(f"cell types: {n_cell_types}")

    print(f"training: {args.epochs} epochs, batch size {args.batch_size}")
    print()

    # load preprocessed data
    print("loading preprocessed data")
    matrix = sio.mmread(os.path.join(args.data_dir, "matrix.mtx")).T.tocsr()

    # load cell type labels
    print("loading cell type labels")
    labels_df = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    cell_type_labels = torch.LongTensor(labels_df['cluster'].values)
    print(f"loaded {len(cell_type_labels)} cell type labels")

    # convert to tensor
    if hasattr(matrix, 'toarray'):
        X = torch.FloatTensor(matrix.toarray())
    else:
        X = torch.FloatTensor(matrix)

    print(f"data shape: {X.shape}")
    print(f"labels shape: {cell_type_labels.shape}")

    # create data loader with actual labels
    dataset = TensorDataset(X, cell_type_labels)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # train GMVAE
    model = train_gmvae(
        data_loader=data_loader,
        input_dim=n_genes,
        n_cell_types=n_cell_types,
        save_path=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    print("\nGMVAE trained")
