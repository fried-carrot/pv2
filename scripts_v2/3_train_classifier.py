#!/usr/bin/env python3
"""
patient level classifier
"""

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np
import pandas as pd
import os
import json
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, f1_score # type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore
import scipy.io as sio

# import GMVAE from script 2
import sys
import os
import importlib.util
spec = importlib.util.spec_from_file_location("train_gmvae", os.path.join(os.path.dirname(__file__), "2_train_gmvae.py"))
train_gmvae_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_gmvae_module)
GMVAE_ZINB = train_gmvae_module.GMVAE_ZINB
Args = train_gmvae_module.Args  # needed for loading checkpoint

# import P4P model
p4p_model_spec = importlib.util.spec_from_file_location("p4p_model", os.path.join(os.path.dirname(__file__), "../repositories/ProtoCell4P/src/model.py"))
p4p_model_module = importlib.util.module_from_spec(p4p_model_spec)
p4p_model_spec.loader.exec_module(p4p_model_module)
P4PModel = p4p_model_module.ProtoCell



class P4PxGMVAE(P4PModel):
    """
    P4P w/ GMVAE embeddings replacing P4P's encoder
    Uses GMVAE-derived z-scores as cell-to-prototype distances
    Enhanced with attention-based pooling (Ilse et al. 2018, Wang et al. 2024)
    """

    def __init__(self, *args, **kwargs):
        super(P4PxGMVAE, self).__init__(*args, **kwargs)
        self.gmvae_model = None

        # attention module for learning cell importance
        self.cell_attention = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def set_gmvae_model(self, gmvae_model):
        """set the frozen GMVAE model for feature extraction"""
        self.gmvae_model = gmvae_model
        self.gmvae_model.eval()

    def forward(self, x, y, ct=None, sparse=True):
        # from: ProtoCell4P/src/model.py lines 64-96
        # Modified: use GMVAE embeddings instead of P4P encoder

        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])

        if sparse:
            x_concat = torch.cat([torch.tensor(x[i].toarray(), dtype=torch.float32) for i in range(len(x))]).to(self.device)
        else:
            x_concat = torch.cat([torch.tensor(x[i], dtype=torch.float32) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)

        # use GMVAE embeddings
        if self.gmvae_model is not None:
            gmvae_z = self.gmvae_model.get_embeddings(x_concat)  # (n_cells, z_dim)
        else:
            # fallback if GMVAE not set
            gmvae_z = self.encode(x_concat)

        import_scores = self.compute_importance(x_concat)  # (n_cell, n_proto, n_class)

        # compute z-scores as cell-to-prototype distances
        if self.gmvae_model is not None and ct is not None:
            # use GMVAE's cell type distributions for z-score calculation
            cell_types_concat = torch.cat([torch.tensor(ct[i]) for i in range(len(ct))]).to(self.device)

            # get global cell type priors
            mu_prior = self.gmvae_model.mu_prior  # (K, z_dim)
            logvar_prior = self.gmvae_model.logvar_prior  # (K, z_dim)
            std_prior = torch.exp(0.5 * logvar_prior)  # (K, z_dim)

            # compute z-scores: (embedding - cell_type_mean) / cell_type_std
            # for each cell, get its cell type's global distribution
            cell_means = mu_prior[cell_types_concat]  # (n_cells, z_dim)
            cell_stds = std_prior[cell_types_concat]  # (n_cells, z_dim)

            # z-score per dimension
            z_scores = (gmvae_z - cell_means) / (cell_stds + 1e-8)  # (n_cells, z_dim)

            # compute distances from z-scores to prototypes
            # use squared distance in z-score space
            c2p_dists = torch.pow(z_scores[:, None, :] - self.prototypes[None, :, :], 2).sum(-1)  # (n_cells, n_proto)
        else:
            # fallback: standard Euclidean distance
            c2p_dists = torch.pow(gmvae_z[:, None] - self.prototypes[None, :], 2).sum(-1)

        # rest of P4P forward pass with attention-weighted pooling
        c_logits = (1 / (c2p_dists+0.5))[:,None,:].matmul(import_scores).squeeze(1)  # (n_cell, n_classes)

        # attention-weighted aggregation per patient (inspired by Ilse et al. 2018)
        logits = []
        for i in range(len(split_idx)-1):
            patient_embeddings = gmvae_z[split_idx[i]:split_idx[i+1]]  # (n_cells_patient, z_dim)
            patient_c_logits = c_logits[split_idx[i]:split_idx[i+1]]  # (n_cells_patient, n_classes)

            # compute attention weights for this patient
            att_scores = self.cell_attention(patient_embeddings)  # (n_cells_patient, 1)
            att_weights = F.softmax(att_scores, dim=0)  # (n_cells_patient, 1)

            # weighted aggregation
            weighted_logit = (patient_c_logits * att_weights).sum(dim=0)  # (n_classes,)
            logits.append(weighted_logit)

        logits = torch.stack(logits)  # (n_patients, n_classes)

        clf_loss = self.ce_(logits, y)

        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf2(import_scores.reshape(-1, self.n_proto * self.n_classes))
            ct_loss = self.ce_(ct_logits, torch.tensor([j for i in ct for j in i]).to(self.device))
        else:
            ct_loss = 0

        # prototype diversity regularization (inspired by Li et al. 2018, Chen et al. 2019)
        # encourages prototypes to capture different patterns
        proto_sim = F.cosine_similarity(
            self.prototypes.unsqueeze(1),
            self.prototypes.unsqueeze(0),
            dim=2
        )
        # exclude diagonal (self-similarity)
        proto_sim = proto_sim * (1 - torch.eye(self.n_proto, device=self.device))

        # diversity loss: penalize high similarity between different prototypes
        diversity_loss = proto_sim.abs().mean()

        total_loss = clf_loss + self.lambda_6 * ct_loss + 0.01 * diversity_loss

        if ct is not None:
            return total_loss, logits, ct_logits
        return total_loss, logits


def load_frozen_gmvae(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # load trained GMVAE
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # override args device with current device
    args = checkpoint['args']
    args.device = device

    # use trained Bulk2SC GMVAE_ZINB model
    model = GMVAE_ZINB(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # restore global priors
    model.mu_prior = checkpoint['mu_prior'].to(device)
    model.logvar_prior = checkpoint['logvar_prior'].to(device)

    # ensure onehot tensor is on correct device
    model.onehot = model.onehot.to(device)

    model.freeze_model()

    return model


def prepare_patient_data(data_dir, device='cpu'):
    """
    prepare patient-level data for P4P framework using real patient structure
    Returns data in P4P's expected format: [patient_cell_matrix, ...], labels, cell_types
    """
    # load preprocessed data
    matrix = sio.mmread(os.path.join(data_dir, "matrix.mtx")).T.tocsr()
    labels_df = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    # group by actual patient IDs
    patient_ids = labels_df['patient_id'].values
    disease_labels = labels_df['disease'].values
    cell_types = labels_df['cluster'].values

    unique_patients = np.unique(patient_ids)
    print(f"found {len(unique_patients)} unique patients")

    # format for P4P: list of sparse matrices per patient
    X_patients = []
    y_patients = []
    ct_patients = []

    for patient_id in unique_patients:
        patient_mask = patient_ids == patient_id
        patient_indices = np.where(patient_mask)[0]

        # patient's raw gene expression
        patient_matrix = matrix[patient_indices, :]
        X_patients.append(patient_matrix)

        # patient's cell types
        patient_cell_types = cell_types[patient_indices]
        ct_patients.append(patient_cell_types)

        # patient's disease label
        patient_disease_label = disease_labels[patient_indices[0]]
        y_patients.append(patient_disease_label)

    y_patients_tensor = torch.tensor(y_patients)

    print(f"prepared {len(X_patients)} patients")
    print(f"cells per patient: min={min(x.shape[0] for x in X_patients)}, max={max(x.shape[0] for x in X_patients)}, mean={np.mean([x.shape[0] for x in X_patients]):.1f}")

    # load metadata
    with open(os.path.join(data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    unique_labels, counts = torch.unique(y_patients_tensor, return_counts=True)
    for label, count in zip(unique_labels, counts):
        disease_name = metadata['diseases'][label.item()]
        print(f"disease {disease_name}: {count} patients")

    return X_patients, y_patients_tensor, ct_patients


def train_classifier(X_patients, y_patients, ct_patients, gmvae_model, n_genes, save_path,
                    epochs=50, learning_rate=1e-4, device='cpu'):
    """
    train P4P classifier
    """
    # load metadata to get P4P configuration
    n_patients = len(X_patients)
    n_classes = len(torch.unique(y_patients))
    n_cell_types = len(set([ct for patient_cts in ct_patients for ct in patient_cts]))

    print(f"training P4P classifier with {n_patients} patients")
    print(f"genes: {n_genes}, cell types: {n_cell_types}, classes: {n_classes}")

    # train/test split
    train_size = int(0.8 * n_patients)
    indices = list(range(n_patients))
    np.random.shuffle(indices)

    X_train = [X_patients[i] for i in indices[:train_size]]
    y_train = y_patients[indices[:train_size]]
    ct_train = [ct_patients[i] for i in indices[:train_size]]

    X_test = [X_patients[i] for i in indices[train_size:]]
    y_test = y_patients[indices[train_size:]]
    ct_test = [ct_patients[i] for i in indices[train_size:]]

    # initialize P4P model with GMVAE modification
    lambdas = {
        "lambda_1": 0,
        "lambda_2": 0,
        "lambda_3": 0,
        "lambda_4": 0,
        "lambda_5": 0,
        "lambda_6": 1
    }
    model = P4PxGMVAE(
        input_dim=n_genes,
        h_dim=128,  # from P4P hyperparameters
        z_dim=64,   # from P4P hyperparameters (must match GMVAE z_dim)
        n_layers=2,
        n_proto=8,  # from P4P hyperparameters
        n_classes=n_classes,
        lambdas=lambdas,  # P4P expects dict, not kwargs
        n_ct=n_cell_types,
        device=device
    ).to(device)

    model.set_gmvae_model(gmvae_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"training on {device}")
    print("training starts\n")

    # P4P training loop
    model.train()
    for epoch in range(epochs):
        print(f"[Epoch {epoch}]")

        optimizer.zero_grad()

        # P4P forward pass with GMVAE z-scores
        total_loss, logits, ct_logits = model(X_train, y_train, ct=ct_train, sparse=True)

        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"avg. train loss: {total_loss.item():.2f}")
        print("")

    print("training Ends\n")

    # P4P evaluation
    model.eval()
    with torch.no_grad():
        test_loss, test_logits, test_ct_logits = model(X_test, y_test, ct=ct_test, sparse=True)

        y_pred = test_logits.argmax(dim=1)
        test_acc = accuracy_score(y_test.cpu(), y_pred.cpu())

        if test_logits.shape[1] == 2:
            test_auc = roc_auc_score(y_test.cpu(), test_logits.cpu()[:,1])
        else:
            test_auc = roc_auc_score(y_test.cpu(), test_logits.cpu(), multi_class="ovo")

        test_f1 = f1_score(y_test.cpu(), y_pred.cpu(), average="macro")

    print(f"test accuracy: {test_acc:.2f} ; ROC AUC score: {test_auc:.2f} ; F1 score: {test_f1:.2f}")

    # save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_genes': n_genes,
        'n_classes': n_classes,
        'test_metrics': {
            'accuracy': test_acc,
            'auc': test_auc,
            'f1': test_f1
        }
    }, save_path)

    print(f"trained classifier saved to: {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train patient classifier')
    parser.add_argument('--data_dir', required=True, help='preprocessed data directory')
    parser.add_argument('--gmvae_model', required=True, help='trained GMVAE model path')
    parser.add_argument('--output', required=True, help='output classifier path')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')

    args = parser.parse_args()

    print("=" * 60)
    print("patient classifier training")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")

    # load frozen GMVAE
    print(f"loading GMVAE from: {args.gmvae_model}")
    gmvae_model = load_frozen_gmvae(args.gmvae_model, device=device)
    print("GMVAE loaded")

    # prepare patient data
    print("\npreparing patient-level data")
    X_patients, y_patients, ct_patients = prepare_patient_data(args.data_dir, device=device)

    # load metadata to get n_genes
    with open(os.path.join(args.data_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    n_genes = metadata['n_genes']

    # train classifier
    print("\ntraining patient classifier")
    model = train_classifier(
        X_patients=X_patients,
        y_patients=y_patients,
        ct_patients=ct_patients,
        gmvae_model=gmvae_model,
        n_genes=n_genes,
        save_path=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    print("\npatient classifier trained")
    print("="*60)
