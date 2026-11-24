import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from torch.nn.functional import mse_loss

from vqtorch.nn import VectorQuant
from vqtorch.nn.gumbel import GumbelVectorQuantizer
from vector_quantize_pytorch import VectorQuantize


class TVQ(pl.LightningModule):

    def __init__(self, 
                 input_dim,
                 codebook_size, 
                 hidden_dim, 
                 commitment_cost,
                 vq_type='vq',
                 ):
        super(TVQ, self).__init__()

        # Initialize networks
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self._init_networks()

        # Initialize vq 
        self.vq_type = vq_type
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost

        self._init_vq()

        # Save hyperparameters

        self.save_hyperparameters()

    ##########################
    # Initialization methods #
    ##########################

    def _init_networks(self):
        # Initialize transformer encoder 
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )

        # Initialize transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.input_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )

        # Initialize FFN before VQ
        self.ffn_before_vq = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Initialize FFN after VQ
        self.ffn_after_vq = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _init_vq(self):
        if self.vq_type == 'vq':
            self.vq = VectorQuant(
                num_embeddings=self.codebook_size,
                embedding_dim=self.hidden_dim,
                commitment_cost=self.commitment_cost
            )
        elif self.vq_type == 'gumbel':
            self.vq = GumbelVectorQuantizer(
                num_embeddings=self.codebook_size,
                embedding_dim=self.hidden_dim,
                commitment_cost=self.commitment_cost
            )
        else:
            raise ValueError(f"Invalid vq_type: {self.vq_type}")

    ####################
    # Forward methods #
    ####################

    def forward(self, x):
        z = self.encoder(x)
        z = self.ffn_before_vq(z)
        z_q, loss = self.vq(z)
        z_q = self.ffn_after_vq(z_q)
        x_hat = self.decoder(z_q)
        return x