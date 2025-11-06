#!/usr/bin/env python3
"""
Uncertainty-Aware Contrastive Multi-Modal Learning for Bulk RNA-seq

Novel approach that:
1. Predicts cellular modalities from bulk RNA-seq with uncertainty quantification
2. Uses contrastive learning to align modalities
3. Uncertainty-weighted fusion for final prediction

Training: Uses scRNA-seq to learn modality predictors
Deployment: Only needs bulk RNA-seq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict


class ModalityPredictor(nn.Module):
    """
    Predicts cellular modalities from bulk RNA-seq with uncertainty.

    Outputs Gaussian distributions: μ, σ for each modality
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 256]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev_dim, output_dim)
        self.logvar_head = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim]
        Returns:
            mu: [batch, output_dim]
            logvar: [batch, output_dim]
        """
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


class ModalityEncoder(nn.Module):
    """Encodes a single modality to embedding space"""
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for multi-modal alignment.

    Positive pairs: different modalities from same patient
    Negative pairs: different modalities from different patients
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of [batch, embedding_dim] tensors (one per modality)
        Returns:
            contrastive_loss: scalar
        """
        batch_size = embeddings[0].size(0)
        n_modalities = len(embeddings)

        # Normalize embeddings
        embeddings = [F.normalize(emb, dim=1) for emb in embeddings]

        total_loss = 0
        n_pairs = 0

        # Compare each pair of modalities
        for i in range(n_modalities):
            for j in range(i + 1, n_modalities):
                # Compute similarity matrix: [batch, batch]
                sim_matrix = torch.matmul(embeddings[i], embeddings[j].T) / self.temperature

                # Positive pairs are on diagonal (same patient, different modality)
                labels = torch.arange(batch_size, device=sim_matrix.device)

                # Symmetric loss
                loss_i = F.cross_entropy(sim_matrix, labels)
                loss_j = F.cross_entropy(sim_matrix.T, labels)

                total_loss += (loss_i + loss_j) / 2
                n_pairs += 1

        return total_loss / n_pairs


class UncertaintyWeightedFusion(nn.Module):
    """
    Fuses modalities weighted by inverse uncertainty.
    High uncertainty → low weight
    """
    def __init__(self, n_modalities: int, embedding_dim: int):
        super().__init__()
        self.n_modalities = n_modalities

        # Learnable temperature for uncertainty scaling
        self.temp = nn.Parameter(torch.ones(1))

    def forward(
        self,
        embeddings: List[torch.Tensor],
        uncertainties: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            embeddings: List of [batch, embedding_dim]
            uncertainties: List of [batch, modality_dim] (variance)
        Returns:
            fused_embedding: [batch, embedding_dim]
        """
        # Average uncertainty per sample per modality
        weights = []
        for unc in uncertainties:
            # Convert variance to weight: 1 / (σ + ε)
            w = 1.0 / (unc.mean(dim=1, keepdim=True) + 1e-6)
            weights.append(w)

        weights = torch.cat(weights, dim=1)  # [batch, n_modalities]
        weights = F.softmax(weights / self.temp, dim=1)  # Normalize

        # Weighted sum
        fused = sum(w.unsqueeze(-1) * emb for w, emb in zip(weights.T, embeddings))

        return fused


class MultiModalBulkPredictor(nn.Module):
    """
    Complete uncertainty-aware multi-modal architecture.

    Predicts 3 modalities from bulk RNA-seq:
    1. Cell type proportions
    2. Cell state distributions (mean embedding per cell type)
    3. Cell-cell communication scores
    """
    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_interactions: int,
        embedding_dim: int = 128,
        state_dim: int = 64,
    ):
        super().__init__()

        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_interactions = n_interactions
        self.embedding_dim = embedding_dim

        # Modality predictors (bulk → modality with uncertainty)
        self.predict_proportions = ModalityPredictor(n_genes, n_cell_types)
        self.predict_states = ModalityPredictor(n_genes, n_cell_types * state_dim)
        self.predict_communication = ModalityPredictor(n_genes, n_interactions)

        # Modality encoders (modality → embedding)
        self.encode_bulk = ModalityEncoder(n_genes, embedding_dim)
        self.encode_proportions = ModalityEncoder(n_cell_types, embedding_dim)
        self.encode_states = ModalityEncoder(n_cell_types * state_dim, embedding_dim)
        self.encode_communication = ModalityEncoder(n_interactions, embedding_dim)

        # Fusion
        self.fusion = UncertaintyWeightedFusion(4, embedding_dim)

        # Losses
        self.contrastive_loss = ContrastiveLoss()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        bulk: torch.Tensor,
        true_proportions: torch.Tensor = None,
        true_states: torch.Tensor = None,
        true_communication: torch.Tensor = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bulk: [batch, n_genes]
            true_proportions: [batch, n_cell_types] (only during training)
            true_states: [batch, n_cell_types * state_dim] (only during training)
            true_communication: [batch, n_interactions] (only during training)
            training: bool

        Returns:
            dict with:
                - embedding: fused multi-modal embedding
                - loss_reconstruction: MSE/KL for modality prediction
                - loss_contrastive: contrastive alignment loss
                - uncertainties: dict of uncertainties per modality
        """
        # Predict modalities with uncertainty
        props_mu, props_logvar = self.predict_proportions(bulk)
        states_mu, states_logvar = self.predict_states(bulk)
        comm_mu, comm_logvar = self.predict_communication(bulk)

        # Sample modalities
        if training:
            props = self.reparameterize(props_mu, props_logvar)
            states = self.reparameterize(states_mu, states_logvar)
            comm = self.reparameterize(comm_mu, comm_logvar)
        else:
            # Use mean during inference
            props = props_mu
            states = states_mu
            comm = comm_mu

        # Normalize proportions to sum to 1
        props = F.softmax(props, dim=1)

        # Encode all modalities
        z_bulk = self.encode_bulk(bulk)
        z_props = self.encode_proportions(props)
        z_states = self.encode_states(states)
        z_comm = self.encode_communication(comm)

        embeddings = [z_bulk, z_props, z_states, z_comm]

        # Compute losses
        losses = {}

        if training and true_proportions is not None:
            # Reconstruction loss
            loss_props = F.mse_loss(props_mu, true_proportions)
            loss_states = F.mse_loss(states_mu, true_states)
            loss_comm = F.mse_loss(comm_mu, true_communication)

            # KL divergence for uncertainty regularization
            kl_props = -0.5 * torch.mean(1 + props_logvar - props_mu.pow(2) - props_logvar.exp())
            kl_states = -0.5 * torch.mean(1 + states_logvar - states_mu.pow(2) - states_logvar.exp())
            kl_comm = -0.5 * torch.mean(1 + comm_logvar - comm_mu.pow(2) - comm_logvar.exp())

            losses['reconstruction'] = loss_props + loss_states + loss_comm
            losses['kl'] = (kl_props + kl_states + kl_comm) * 0.1  # Weight KL lower

            # Contrastive loss
            losses['contrastive'] = self.contrastive_loss(embeddings)

        # Uncertainty-weighted fusion
        uncertainties = [
            torch.exp(props_logvar),
            torch.exp(states_logvar),
            torch.exp(comm_logvar)
        ]

        # Use predicted modality embeddings (not bulk) for fusion
        fused_embedding = self.fusion(
            [z_props, z_states, z_comm],
            uncertainties
        )

        return {
            'embedding': fused_embedding,
            'losses': losses,
            'uncertainties': {
                'proportions': uncertainties[0].mean(),
                'states': uncertainties[1].mean(),
                'communication': uncertainties[2].mean()
            },
            'modalities': {
                'proportions': props,
                'states': states,
                'communication': comm
            }
        }


class MultiModalClassifier(nn.Module):
    """
    Complete patient classifier with multi-modal bulk predictor.
    """
    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_interactions: int,
        n_classes: int,
        embedding_dim: int = 128,
        state_dim: int = 64,
    ):
        super().__init__()

        self.multimodal = MultiModalBulkPredictor(
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            n_interactions=n_interactions,
            embedding_dim=embedding_dim,
            state_dim=state_dim
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(
        self,
        bulk: torch.Tensor,
        labels: torch.Tensor = None,
        true_proportions: torch.Tensor = None,
        true_states: torch.Tensor = None,
        true_communication: torch.Tensor = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bulk: [batch, n_genes]
            labels: [batch] (patient labels)
            true_proportions/states/communication: ground truth from scRNA-seq
            training: bool

        Returns:
            dict with logits, losses, uncertainties
        """
        # Get multi-modal embedding
        mm_output = self.multimodal(
            bulk=bulk,
            true_proportions=true_proportions,
            true_states=true_states,
            true_communication=true_communication,
            training=training
        )

        # Classify
        logits = self.classifier(mm_output['embedding'])

        output = {
            'logits': logits,
            'uncertainties': mm_output['uncertainties'],
            'modalities': mm_output['modalities']
        }

        # Compute total loss
        if training and labels is not None:
            loss_cls = F.cross_entropy(logits, labels)

            total_loss = (
                loss_cls +
                mm_output['losses']['reconstruction'] * 0.5 +
                mm_output['losses']['kl'] * 0.1 +
                mm_output['losses']['contrastive'] * 0.3
            )

            output['loss'] = total_loss
            output['loss_components'] = {
                'classification': loss_cls.item(),
                'reconstruction': mm_output['losses']['reconstruction'].item(),
                'kl': mm_output['losses']['kl'].item(),
                'contrastive': mm_output['losses']['contrastive'].item()
            }

        return output


if __name__ == "__main__":
    # Test the architecture
    batch_size = 16
    n_genes = 1000
    n_cell_types = 8
    n_interactions = 50
    n_classes = 2

    model = MultiModalClassifier(
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        n_interactions=n_interactions,
        n_classes=n_classes
    )

    # Fake data
    bulk = torch.randn(batch_size, n_genes)
    labels = torch.randint(0, n_classes, (batch_size,))
    true_props = torch.randn(batch_size, n_cell_types).softmax(dim=1)
    true_states = torch.randn(batch_size, n_cell_types * 64)
    true_comm = torch.randn(batch_size, n_interactions)

    # Forward pass
    output = model(
        bulk=bulk,
        labels=labels,
        true_proportions=true_props,
        true_states=true_states,
        true_communication=true_comm,
        training=True
    )

    print("Output keys:", output.keys())
    print("Logits shape:", output['logits'].shape)
    print("Total loss:", output['loss'].item())
    print("Loss components:", output['loss_components'])
    print("Uncertainties:", output['uncertainties'])
    print("\nArchitecture test passed!")
