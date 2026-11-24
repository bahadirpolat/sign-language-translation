import os
import h5py
import wandb
import warnings
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# VQ modules
from vqtorch.nn import VectorQuant
from vector_quantize_pytorch import VectorQuantize, LFQ
# Losses
from info_nce import InfoNCE
from torch.nn import TripletMarginWithDistanceLoss, CosineEmbeddingLoss
from tqdm.auto import tqdm


def collate_fn(batch):
    anchor = [item[0] for item in batch]
    positive = [item[1] for item in batch]
    negative = [item[2] for item in batch]

    anchor = pad_sequence(anchor, batch_first=True, padding_value=-100)
    positive = pad_sequence(positive, batch_first=True, padding_value=-100)
    negative = pad_sequence(negative, batch_first=True, padding_value=-100)

    return anchor, positive, negative

class CLVQ(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_heads,
        codebook_size,
        commitment_cost,
        commit_coef=0.2,
        margin=0.5,
        num_training_samples=10_000,
        freeze_codebook=True,
        use_wandb=False,
        batch_size=32,
        vq_type="vq-ste++",
        loss_type="triplet",
        dist_fn="cosine",
        rotation_trick=False,
        learning_rate=1e-3,
        post_linear=False,
        save_dir="models/base-cvq",
    ):
        """
        CLIP with VQ-VAE model
        """
        super().__init__()

        # Model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.codebook_size = codebook_size
        self.post_linear = post_linear

        self._init_encoder()

        # Loss
        self.margin = margin
        self.dist_fn = dist_fn
        self.loss_type = loss_type
        self.commitment_cost = commitment_cost

        self._init_loss()

        # Vector quantization
        self.vq_type = vq_type
        self.freeze_codebook = freeze_codebook
        self._init_vq()

        # Training parameters
        self.learning_rate = learning_rate
        self.commit_coef = commit_coef

        # Create step to save validation batches
        self.state = {"val_embeddings": [], "val_indices": []}

        # Log hyperparameters
        self.save_dir = save_dir 
        self.use_wandb = use_wandb
        self.save_hyperparameters()

    #########################
    # Initialization functions
    #########################

    def _init_encoder(self):
        # Model parameters
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.GELU()
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads),
            num_layers=self.num_layers,
        )

        if self.post_linear:
            self.post_mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))

    def _init_vq(self):
        if self.vq_type == "vanilla":
            self.vq = VectorQuantize(
                dim=self.hidden_dim,
                codebook_dim=self.hidden_dim,
                codebook_size=self.codebook_size,
                rotation_trick=self.rotation_trick,
                kmeans_init=True,
                kmeans_iters=100,
                stochastic_sample_codes=True,
            )
        # elif self.vq_type == 'gumbel':
        #     self.vq = GumbelVectorQuantizer(
        #         input_dim=self.hidden_dim,
        #         num_vars=self.codebook_size,
        #         groups=1,
        #         # temp_tuple=(1.0,),
        #         vq_dim=self.hidden_dim,
        #     )
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
                dim=-1,
            )
        elif self.vq_type == "lfq":
            self.vq = LFQ(
                dim=self.hidden_dim,
                codebook_size=self.codebook_size,
                commitment_loss_weight=self.commitment_cost,
                entropy_loss_weight=0.9,
            )
        # elif self.vq_type == "rp":
        #     self.vq = RandomProjectionQuantizer(
        #         input_dim=self.hidden_dim,
        #         cb_dim=self.hidden_dim,
        #         cb_vocab=self.codebook_size,
        #     )
        else:
            raise ValueError(f"VQ type {self.vq_type} not supported")
        
        # Freeze the codebook
        if self.freeze_codebook:
            self.vq.codebook.requires_grad = False

    def _init_loss(self):
        if self.loss_type == "triplet":
            self.loss = TripletMarginWithDistanceLoss(
                margin=self.margin,
                reduction="sum",
                distance_function=lambda x, y: (
                    F.cosine_similarity(x, y)
                    if self.dist_fn == "cosine"
                    else F.pairwise_distance(x, y)
                ),
            )

        if self.loss_type == "cosine":
            self.loss = CosineEmbeddingLoss(margin=self.margin, reduce="sum")
            if self.dist_fn == "euclidean":
                warnings.warn(
                    "Cosine loss is not supported with euclidean distance. Switching to cosine distance."
                )
                self.dist_fn = "cosine"

        if self.loss_type == "info-nce":
            self.loss = InfoNCE(
                temperature=0.7,
                negative_mode="unpaired",
                reduction="sum",
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    #########################
    # Forward functions
    #########################

    def forward_encoder(self, x, quantize=False):
        x = self.mlp(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        if quantize:
            if self.vq_type == "vq-ste++":
                quantized, vq_dict = self.vq(x)
                commit_loss = vq_dict["loss"]
                tokens = vq_dict["q"]
            else:
                quantized, tokens, commit_loss = self.vq(x)
                quantized = quantized.squeeze()
            return quantized, commit_loss, tokens

        if self.post_linear:
            x = self.post_mlp(x)

        return x

    def forward(self, anchor, positive, negative):
        z1, loss, _ = self.forward_encoder(anchor, quantize=True)
        z2 = self.forward_encoder(positive)
        z3 = self.forward_encoder(negative)
        return z1, z2, z3, loss

    def quantize(self, x):
        z, _, tokens = self.forward_encoder(x, quantize=True)
        return tokens.squeeze()

    def _calculate_loss(self, z1, z2, z3):
        if self.loss_type == "triplet":
            loss = self.loss(z1, z2, z3)

        if self.loss_type == "cosine":
            pos_loss = self.loss(z1, z2, torch.ones((z1.size(0))).cuda())
            neg_loss = self.loss(z1, z3, -1 * torch.ones((z1.size(0))).cuda())
            neg_loss = neg_loss + self.loss(
                z2, z3, -1 * torch.ones((z1.size(0))).cuda()
            )
            loss = pos_loss + neg_loss

        if self.loss_type == "info-nce":
            loss = self.loss(z1, z2, z3)

        return loss
    #########################
    # Training functions
    #########################

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        z1, z2, z3, commit_loss = self(anchor, positive, negative)

        loss = self._calculate_loss(z1, z2, z3)

        self._log_embedding_similarity(z1, z2, z3, mode="train")
        self.log("train_contrastive_loss", loss)
        self.log("train_commit_loss", commit_loss)
        loss = loss + self.commit_coef * commit_loss

        return loss

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        z1, z2, z3, commit_loss = self(anchor, positive, negative)

        if self.loss_type == "triplet":
            loss = self.loss(z1, z2, z3)

        if self.loss_type == "cosine":
            pos_loss = self.loss(z1, z2, torch.ones((z1.size(0))).cuda())
            neg_loss = self.loss(z1, z3, -1 * torch.ones((z1.size(0))).cuda())
            neg_loss = neg_loss + self.loss(
                z2, z3, -1 * torch.ones((z1.size(0))).cuda()
            )
            self.log("val_pos_loss", pos_loss)
            self.log("val_neg_loss", neg_loss)
            loss = pos_loss + neg_loss

        if self.loss_type == "info-nce":
            loss = self.loss(z1, z2, z3)

        self.state["val_embeddings"].append((z1, z2, z3))

        self._log_embedding_similarity(z1, z2, z3, mode="val")
        self.log("val_contrastive_loss", loss)
        self.log("val_commit_loss", commit_loss)
        return loss
    
    def on_train_epoch_end(self):
        if self.use_wandb:
            self._log_embedding_distribution()

    def on_validation_end(self):
        # Visualize the validation embeddings

        import matplotlib.pyplot as plt
        import numpy as np

        # print(
        #     f"Visualizing the embeddings for the validation set with {len(self.state['val_embeddings'])} samples",
        #     f'Each z1 sample has shape: {self.state["val_embeddings"][0][0].shape}',
        #     f'Each z2 sample has shape: {self.state["val_embeddings"][0][1].shape}',
        #     f'Each z3 sample has shape: {self.state["val_embeddings"][0][2].shape}',
        #     sep="\n",
        # )

        # Merge the embeddings
        z1s = np.concatenate([
            x[0].cpu().detach().numpy() for x in self.state["val_embeddings"]
        ], axis=0).squeeze()
        z2s = np.concatenate([
            x[1].cpu().detach().numpy() for x in self.state["val_embeddings"]
        ], axis=0).squeeze()
        z3s = np.concatenate([
            x[2].cpu().detach().numpy() for x in self.state["val_embeddings"]
        ], axis=0).squeeze()

        z_all = np.concatenate([z1s, z2s, z3s], axis=0)

        # print('** Z1 shape:', z1s.shape)
        # print('** Embed shapes:', z_all.shape)

        # Perform PCA
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        z_all_pca = pca.fit_transform(z_all)

        # Plot the embeddings
        fig, ax = plt.subplots()
        anchors = ax.scatter(z_all_pca[: len(z1s), 0], z_all_pca[: len(z1s), 1], c="r")
        positives = ax.scatter(
            z_all_pca[len(z1s) : 2 * len(z1s), 0], z_all_pca[len(z1s) : 2 * len(z1s), 1], c="g"
        )
        negatives = ax.scatter(
            z_all_pca[2 * len(z1s) :, 0], z_all_pca[2 * len(z1s) :, 1], c="b"
        )
        ax.legend([anchors, positives, negatives], ["Anchor", "Positive", "Negative"])
        if self.use_wandb:
            self.logger.experiment.log({"sample_distribution": wandb.Image(fig)})

        plt.savefig(os.path.join(self.save_dir, "sample_distribution.png"))
        
        plt.close(fig)

    #########################
    # Logging functions
    #########################

    def _log_embedding_similarity(self, z1, z2, z3, mode="train"):
        if self.dist_fn == "euclidean":
            dist_pos = F.pairwise_distance(z1, z2).mean()
            dist_neg = F.pairwise_distance(z1, z3).mean()
            dist_aug = F.pairwise_distance(z2, z3).mean()
            self.log(f"{mode}_dist_pos1_pos1_aug", dist_pos)
            self.log(f"{mode}_dist_pos1_pos2", dist_neg)
            self.log(f"{mode}_dist_pos1_aug_pos2", dist_aug)

        if self.dist_fn == "cosine":
            dist_pos = F.cosine_similarity(z1, z2).mean()
            dist_neg = F.cosine_similarity(z1, z3).mean()
            dist_aug = F.cosine_similarity(z2, z3).mean()
            self.log(f"{mode}_dist_pos1_pos1_aug", dist_pos)
            self.log(f"{mode}_dist_pos1_pos2", dist_neg)
            self.log(f"{mode}_dist_pos1_aug_pos2", dist_aug)

    def _log_embedding_distribution(self):
        with torch.no_grad():
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt

            if self.vq_type == "vanilla":
                codebook = self.vq.codebook.cpu().detach().numpy()
            if self.vq_type == "vq-ste++":
                codebook = self.vq.codebook.weight.cpu().detach().numpy()

            pca = PCA(n_components=2)
            codebook_pca = pca.fit_transform(codebook)

            fig, ax = plt.subplots()
            ax.scatter(codebook_pca[:, 0], codebook_pca[:, 1])
            if self.use_wandb:
                self.logger.experiment.log({"codebook_pca": wandb.Image(fig)})
            else:
                plt.savefig(os.path.join(self.save_dir, "codebook_pca.png"))

            plt.close(fig)

    #########################
    # Inference functions
    #########################

    def infer_sequence(self, pose, stride=5, window_size=20):
        """
        Infer the tokens for a sequence of poses

        Args:
            pose (np.array): Sequence of poses
            stride (int): Stride to use for inference
            window_size (int): Window size to use for inference

        Returns:
            torch.tensor: tokens for the sequence
        """
        window_batches = []
        for j in range(0, len(pose), stride):
            if j + window_size < len(pose):
                window_batch = pose[j : j + window_size]
            else:
                window_batch = pose[j:]
            window_batches.append(torch.tensor(window_batch))

        window_batches = pad_sequence(
            window_batches, batch_first=True, padding_value=-100
        )
        tokens = self.quantize(window_batches.cuda())
        return tokens

    def infer_dataset(
        self, dataset, save_dir, windows=[20, 30, 40], strides=[5, 10, 15], mode="train"
    ):
        """
        Infer the dataset and save the tokens to a h5 file

        Args:
            dataset (torch.utils.data.Dataset): Dataset to infer
            save_dir (str): Path to save the h5 file
            windows (list): List of window sizes to use for inference
            strides (list): List of strides to use for inference
            mode (str): Mode of the dataset

        Returns:
            None
        """
        print(
            f"Infering {mode} dataset...",
            "with windows:",
            windows,
            "and strides:",
            strides,
        )

        if os.path.exists(save_dir):
            os.remove(save_dir)

        with torch.no_grad():
            self.eval()
            with h5py.File(save_dir, "a") as f:
                for i in tqdm(range(len(dataset)), desc=f"Infering {mode} dataset"):
                    file_id, pose, sentence = dataset[i]
                    tokens = self.infer_sequence(pose)
                    f.create_group(file_id)
                    f[file_id].create_dataset("features", data=tokens.cpu().numpy())
                    f[file_id].create_dataset("sentence", data=sentence)

                    for stride in strides:
                        for window_size in windows:
                            tokens = self.infer_sequence(
                                pose, stride=stride, window_size=window_size
                            )
                            new_id = f"{file_id}.{stride}.{window_size}"
                            f.create_group(new_id)
                            f[new_id].create_dataset(
                                "features", data=tokens.cpu().numpy()
                            )
                            f[new_id].create_dataset("sentence", data=sentence)

    def get_comparative_results(self, dataset, save_dir, mode="train"):
        self.eval()
        tokens = []
        for i in range(len(dataset)):
            pos_1, pos_1_aug, pos_2, lex1, lex2 = dataset[i]

            pos_1 = pos_1.cuda() if torch.cuda.is_available() else pos_1
            pos_2 = pos_2.cuda() if torch.cuda.is_available() else pos_2
            pos_1_aug = pos_1_aug.cuda() if torch.cuda.is_available() else pos_1_aug

            pos_1 = pos_1.unsqueeze(0)
            pos_2 = pos_2.unsqueeze(0)
            pos_1_aug = pos_1_aug.unsqueeze(0)

            pos_1_tokens = self.quantize(pos_1)
            pos_2_tokens = self.quantize(pos_2)
            pos_1_aug_tokens = self.quantize(pos_1_aug)

            tokens.append(
                {
                    "pos_1": pos_1_tokens.tolist(),
                    "pos_1_aug": pos_1_aug_tokens.tolist(),
                    "pos_2": pos_2_tokens.tolist(),
                    "lex1": lex1,
                    "lex2": lex2,
                }
            )

        os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)
        df = pd.DataFrame(tokens)
        df.to_csv(
            os.path.join(
                save_dir, "results", f"{dataset.dataset_id}.test.constrastive.csv"
            ),
            index=False,
        )
