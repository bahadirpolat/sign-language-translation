import os
import h5py
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from vqtorch.nn import VectorQuant
from vector_quantize_pytorch import VectorQuantize
from sklearn.decomposition import PCA

# Change matplotlib backend to Agg
plt.switch_backend("Agg")


class VQVAE(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        codebook_size,
        hidden_dim,
        commitment_cost,
        vq_type="vq",
        use_wandb=False,
        save_dir=None,
        num_groups=2,
        loss_reduction="mean",
        codebook_diversity_loss_weight=0.5,
    ):
        super(VQVAE, self).__init__()

        # Initialize networks
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self._init_networks()

        # Initialize vq
        self.vq_type = vq_type
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.num_groups = num_groups
        self.codebook_diversity_loss_weight = codebook_diversity_loss_weight
        self.loss_reduction = loss_reduction

        self._init_vq()

        # Save hyperparameters
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.state = {
            "epochs": 0,
            "train_loss": [],
            "val_loss": [],
        }
        self.save_hyperparameters()

    ##########################
    # Initialization methods #
    ##########################

    def _init_networks(self):
        # Initialize encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),  # Assuming input images are 28x28
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.hidden_dim),
        )

        # Initialize decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.input_dim),
        )

    def _init_vq(self):
        # Initialize VQ layer
        if self.vq_type == "vanilla":
            self.vq = VectorQuantize(
                dim=self.hidden_dim,
                codebook_size=self.codebook_size,
                commitment_weight=self.commitment_cost,
                # kmeans_init=True,
                # kmeans_iters=100,
                # layernorm_after_project_in=True,
                # sync_kmeans=True,
                codebook_diversity_loss_weight=0.5,
                orthogonal_reg_weight=0.9,
                learnable_codebook=True, ema_update=False,
                threshold_ema_dead_code=20,
            )
        elif self.vq_type == "group-vq":
            import math

            codebook_size = (
                math.ceil(math.log10(self.codebook_size) / math.log10(2))
                / self.num_groups
            )
            codebook_size = 2 ** int(codebook_size)
            print(f"Codebook size per group: {codebook_size}")
            self.vq = VectorQuantize(
                dim=self.hidden_dim // self.num_groups,
                codebook_size=codebook_size,
                commitment_weight=self.commitment_cost,
                kmeans_init=True,
                kmeans_iters=100,
                layernorm_after_project_in=True,
                sync_kmeans=True,
                codebook_diversity_loss_weight=0.7,
                orthogonal_reg_weight=0.9,
                learnable_codebook=True,
                threshold_ema_dead_code=20,
            )
        elif self.vq_type == "vq-ste++":
            self.vq = VectorQuant(
                feature_size=self.hidden_dim,
                num_codes=self.codebook_size,
                beta=0.98,
                kmeans_init=True,
                norm=None,
                cb_norm=None,
                affine_lr=10.0,
                sync_nu=0.2,
                replace_freq=20,
                dim=-1
            )
        elif self.vq_type == "group-vq-ste++":
            import math

            codebook_size = (
                math.ceil(math.log10(self.codebook_size) / math.log10(2))
                / self.num_groups
            )
            codebook_size = 2 ** int(codebook_size)
            print(f"Codebook size per group: {codebook_size}")
            self.vq = VectorQuant(
                feature_size=self.hidden_dim // self.num_groups,
                num_codes=codebook_size,
                beta=0.98,
                kmeans_init=True,
                norm=None,
                cb_norm=None,
                affine_lr=10.0,
                sync_nu=0.2,
                replace_freq=20,
                dim=-1,
            )
        else:
            raise ValueError(f"VQ type {self.vq_type} not supported")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    ###################
    # Forward methods #
    ###################

    def forward(self, x):

        z = self.encoder(x)

        if self.vq_type == "vanilla":
            quantized, indices, loss = self.vq(z)
        if self.vq_type == "vq-ste++":
            quantized, vq_dict = self.vq(z)
            indices = vq_dict["q"]
            loss = vq_dict["loss"]
        if self.vq_type == "group-vq":
            quantized = torch.zeros_like(z, device=z.device)
            # indices are (B, G)
            indices = torch.zeros((z.size(0), self.num_groups), device=z.device)
            loss = 0
            # Reshape z (B, D) -> (B, G, D/G)
            for group in range(self.num_groups):
                z_group = z[
                    :,
                    group
                    * self.hidden_dim
                    // self.num_groups : (group + 1)
                    * self.hidden_dim
                    // self.num_groups,
                ]
                z_group.to(z.device)
                quantized_group, indices_group, loss_group = self.vq(z_group)
                quantized[
                    :,
                    group
                    * self.hidden_dim
                    // self.num_groups : (group + 1)
                    * self.hidden_dim
                    // self.num_groups,
                ] = quantized_group
                indices[:, group] = indices_group
                loss += loss_group
        if self.vq_type == "group-vq-ste++":
            quantized = torch.zeros_like(z, device=z.device)
            indices = torch.zeros((z.size(0), self.num_groups), device=z.device)
            loss = 0
            for group in range(self.num_groups):
                z_group = z[
                    :,
                    group
                    * self.hidden_dim
                    // self.num_groups : (group + 1)
                    * self.hidden_dim
                    // self.num_groups,
                ]
                z_group.to(z.device)
                quantized_group, vq_dict = self.vq(z_group)
                indices_group = vq_dict["q"].reshape(-1)
                quantized[
                    :,
                    group
                    * self.hidden_dim
                    // self.num_groups : (group + 1)
                    * self.hidden_dim
                    // self.num_groups,
                ] = quantized_group
                indices[:, group] = indices_group
                loss += vq_dict["loss"]

        x_recon = self.decoder(quantized)

        return loss, x_recon, indices

    def quantize(self, x):
        z = self.encoder(x)
        if self.vq_type == "vanilla":
            _, indices, _ = self.vq(z)
        if self.vq_type == "vq-ste++":
            _, vq_dict = self.vq(z)
            indices = vq_dict["q"]
        if self.vq_type == "group-vq":
            # merge indices from all groups
            indices = torch.zeros_like(z)
            for group in range(self.num_groups):
                z_group = z[
                    :,
                    group
                    * self.hidden_dim
                    // self.num_groups : (group + 1)
                    * self.hidden_dim
                    // self.num_groups,
                ]
                z_group.to(z.device)
                _, indices_group = self.vq(z_group)
                indices[ :, group ] = indices_group          
        if self.vq_type == "group-vq-ste++":
            indices = torch.zeros_like(z)
            for group in range(self.num_groups):
                z_group = z[
                    :,
                    group
                    * self.hidden_dim
                    // self.num_groups : (group + 1)
                    * self.hidden_dim
                    // self.num_groups,
                ]
                z_group.to(z.device)
                _, vq_dict = self.vq(z_group)
                indices_group = vq_dict["q"].reshape(-1)
                indices[ :, group ] = indices_group
                    
        return indices

    ####################
    # Training methods #
    ####################

    def training_step(self, batch, batch_idx):
        x = batch
        loss, x_recon, perplexity = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x, reduction=self.loss_reduction)
        self.log("train_loss", loss + recon_loss)

        return loss + recon_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss, x_recon, perplexity = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x, reduction=self.loss_reduction)
        self.log("val_loss", loss + recon_loss)

        return loss + recon_loss

    def on_train_epoch_end(self):
        self._log_embedding_distribution()
        self._log_codebook_reconstruction()
        self.state["epochs"] += 1

    ####################
    # Utility methods  #
    ####################

    def infer_dataset(self, dataset, save_dir, mode="train"):

        with torch.no_grad():
            with h5py.File(save_dir, "a") as f:
                for idx, (file_id, x, sentence) in enumerate(
                    tqdm(dataset), desc=f"Infering {mode} dataset"
                ):
                    x = x.unsqueeze(0)
                    indices = self.quantize(x)

                    f.create_group(file_id)
                    f[file_id].create_dataset("features", data=indices.cpu().numpy())
                    f[file_id].create_dataset("sentence", data=sentence)

    def _log_embedding_distribution(self):
        with torch.no_grad():
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt

            codebooks = None
            if self.vq_type == "vanilla" or self.vq_type == "group-vq":
                codebook = self.vq.codebook.cpu().detach().numpy()
            elif self.vq_type == "vq-ste++" or self.vq_type == "group-vq-ste++":
                codebook = self.vq.codebook.weight.cpu().detach().numpy()

            if codebooks is None:
                codebooks = [codebook]

            for idx, codebook in enumerate(codebooks):
                print(f"Codebook {idx}")
                # check if codebook is has nan values
                if np.isnan(codebook).any():
                    print("Codebook has NaN values")
                    continue

                pca = PCA(n_components=2)
                codebook_pca = pca.fit_transform(codebook)

                fig, ax = plt.subplots()
                ax.scatter(codebook_pca[:, 0], codebook_pca[:, 1])
                if self.use_wandb:
                    self.logger.experiment.log(
                        {f"codebook_pca_{idx}": wandb.Image(fig)}
                    )
                
                plt.savefig(os.path.join(self.save_dir, f"codebook_pca_{idx}.png"))
                plt.close(fig)

    def _log_codebook_reconstruction(self):

        # Decode all codebook entries and visualize keypoints with mediapipe
        from signpiece.datasets import POSE_LANDMARKS

        POSE_CONNECTIONS = [
            (0, 1),
            (0, 4),
            (1, 2),
            (2, 3),
            (3, 7),
            (4, 5),
            (5, 6),
            (6, 8),
            (9, 10),
            (11, 12),
            (11, 13),
            (11, 23),
            (12, 14),
            (12, 24),
            (13, 15),
            (14, 16),
            (15, 17),
            (15, 19),
            (15, 21),
            (16, 18),
            (16, 20),
            (16, 22),
            (17, 19),
            (18, 20),
            (23, 24),
            (23, 25),
            (24, 26),
            (25, 27),
            (26, 28),
            (27, 29),
            (27, 31),
            (28, 30),
            (28, 32),
            (29, 31),
            (30, 32),
        ]
        HAND_CONNECTIONS = [
            (0, 1),
            (0, 5),
            (0, 17),
            (1, 2),
            (2, 3),
            (3, 4),
            (5, 6),
            (5, 9),
            (6, 7),
            (7, 8),
            (9, 10),
            (9, 13),
            (10, 11),
            (11, 12),
            (13, 14),
            (13, 17),
            (14, 15),
            (15, 16),
            (17, 18),
            (18, 19),
            (19, 20),
        ]

        save_path = os.path.join(self.save_dir, "codebook_reconstruction")
        os.makedirs(save_path, exist_ok=True)

        # # delete all files in the directory
        # for file in os.listdir(save_path):
        #     os.remove(os.path.join(save_path, file))

        with torch.no_grad():
            if self.vq_type == "vanilla" or self.vq_type == "group-vq":
                codebook = self.vq.codebook
            if self.vq_type == "vq-ste++" or self.vq_type == "group-vq-ste++":
                codebook = self.vq.codebook.weight.data

            max_display = 10
            fig, axs = plt.subplots(max_display, 3, figsize=(15, 5 * max_display))

            for c_idx in tqdm(range(max_display), desc="Visualizing codebook"):

                if self.vq_type == "vanilla" or self.vq_type == "vq-ste++":
                    idx = np.random.randint(0, self.codebook_size)
                    code_encoding = codebook[idx].unsqueeze(0)
                elif self.vq_type == "group-vq" or self.vq_type == "group-vq-ste++":
                    code_encoding = torch.zeros(self.hidden_dim)
                    idx = []
                    for i in range(self.num_groups):
                        _idx = np.random.randint(0, codebook.size(0))
                        idx += [_idx]
                        code_encoding[
                            i
                            * self.hidden_dim
                            // self.num_groups : (i + 1)
                            * self.hidden_dim
                            // self.num_groups
                        ] = codebook[_idx]
                    idx = "-".join([str(i) for i in idx])

                # Code encoding device
                code_encoding = torch.tensor(code_encoding).to(self.device)

                keypoints = self.decoder(code_encoding)

                keypoints = keypoints.cpu().numpy()

                keypoints = keypoints.reshape(-1, 3)
                # FIXME: This is a hack to make the keypoints visible in the plot
                pose = keypoints[:8, :]
                left_hand = keypoints[8 : 21 + 8, :]
                right_hand = keypoints[21 + 8 :, :]

                # Visualize keypoints with mediapipe connections
                # Create subplot for each body part

                # Pose
                for connection in POSE_CONNECTIONS:
                    if (
                        connection[0] not in POSE_LANDMARKS
                        or connection[1] not in POSE_LANDMARKS
                    ):
                        continue
                    pose_connection_1 = POSE_LANDMARKS.index(connection[0])
                    pose_connection_2 = POSE_LANDMARKS.index(connection[1])
                    start = pose[pose_connection_1] * -1
                    end = pose[pose_connection_2] * -1
                    axs[c_idx][0].plot([start[0], end[0]], [start[1], end[1]], "r")

                axs[c_idx][0].axis("off")
                axs[c_idx][0].set_title(f"Pose {idx}")

                # Right hand
                for connection in HAND_CONNECTIONS:
                    start = right_hand[connection[0]] * -1
                    end = right_hand[connection[1]] * -1

                    axs[c_idx][1].plot([start[0], end[0]], [start[1], end[1]], "g")

                axs[c_idx][1].axis("off")
                axs[c_idx][1].set_title(f"Right Hand {idx}")

                # Left hand
                for connection in HAND_CONNECTIONS:
                    start = left_hand[connection[0]] * -1
                    end = left_hand[connection[1]] * -1

                    axs[c_idx][2].plot([start[0], end[0]], [start[1], end[1]], "b")

                axs[c_idx][2].axis("off")
                axs[c_idx][2].set_title(f"Left Hand {idx}")

            # Save figure as image
            plt.savefig(os.path.join(save_path, f"codebooks.png"))

            # Log figure to wandb Image
            if self.use_wandb:
                self.logger.experiment.log({f"codebook-recon": wandb.Image(fig)})
