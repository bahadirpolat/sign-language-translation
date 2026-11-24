import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import h5py
import pickle
from abc import abstractmethod
from tqdm.notebook import tqdm
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from typing import Tuple, List, Dict, Any, Optional

# from signpiece.transforms import SignPieceTransforms
import mediapipe as mp

import logging

logging.basicConfig(level=logging.INFO)
logging.BASIC_FORMAT = "%(asctime)s - %(message)s"

logger = logging.getLogger(name="[KMeansCharTokenizer]")

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

POSE_CONNECTION_OPENPOSE = [
    connection
    for connection in POSE_CONNECTIONS
    if connection[0] < 24 and connection[1] < 24
]

# Implement TokenizerConfig
@dataclass
class TokenizerConfig:
    """
    TokenizerConfig is a dataclass that contains the configuration of the tokenizer.
    """

    multicue: bool
    num_codebooks: Optional[int]
    left_hand_codebooks: Optional[int]
    right_hand_codebooks: Optional[int]
    pose_codebooks: Optional[int]
    temporal_window: int
    normalizations: List[str]
    dataset_id: Optional[str]
    training_arguments: Dict[str, Optional[int]]


BASE_CONFIG = TokenizerConfig(
    multicue=False,
    num_codebooks=None,
    left_hand_codebooks=None,
    right_hand_codebooks=None,
    pose_codebooks=None,
    temporal_window=1,
    normalizations=["none"],
    dataset_id=None,
    training_arguments={"n_iters": None, "percentage": None, "batch_size": None},
)

class BaseQuantizer:
    """
    BaseQuantizer is the base class for all tokenizers. It provides the basic functionality such as
    tokenization, training, saving, loading, and visualization of the tokens.

    
    # Interace for Tokenizers
    # - Tokenize
    # - Train
    # - Save
    # - Load
    # - Visualize
    # - Infer

    # IDEA: What if we calculate the KMeans for each cue separately?
    #      This way we can have a more fine-grained representation of the data
    #      and we can also have a more interpretable representation of the data
    #      since we can see which clusters are more common for each cue


    # IDEA: Can we measure distance only with appropriately detected keypoints?
    #      The formulation can be \sum_
    # IDEA: This is a naive approach, we should consider the temporal window

    FIXME: Do not hardcode the number of keypoints in the tokenizer.
    FIXME: Dont hardcode the number of keypoints
    TODO: Export token visualizations as well
    TODO: Add enum for dataset_id and dataset_type

    Attributes:
    - multicue: If True, the tokenizer will tokenize the data using multiple KMeans models.
    - num_codebooks: The number of codes for the singlecue tokenizer.
    - left_hand_codebooks: The number of codes for the left hand.
    - right_hand_codebooks: The number of codes for the right hand.
    - pose_codebooks: The number of codes for the pose.
    - temporal_window: The number of frames to consider for the tokenization.
    - normalize: If True, the data will be normalized before tokenization.
    """

    def __init__(
        self,
        model_id: str,
        multicue: bool,
        num_codebooks: Optional[int],
        left_hand_codebooks: Optional[int],
        right_hand_codebooks: Optional[int],
        pose_codebooks: Optional[int],
        temporal_window: int,
        normalizations: List[str] = None
    ) -> None:

        self.model_id = model_id

        self.multicue = multicue
        self.num_codebooks = num_codebooks
        self.left_hand_codebooks = left_hand_codebooks
        self.right_hand_codebooks = right_hand_codebooks
        self.pose_codebooks = pose_codebooks

        self.all_km = None
        self.left_hand_km = None
        self.right_hand_km = None
        self.pose_km = None

        self.normalizations = normalizations

        self.config = {
            "multicue": self.multicue,
            "num_codebooks": self.num_codebooks,
            "left_hand_codebooks": self.left_hand_codebooks,
            "right_hand_codebooks": self.right_hand_codebooks,
            "pose_codebooks": self.pose_codebooks,
            "temporal_window": temporal_window,
            "normalizations": normalizations,
            "dataset_id": None,
            "training_arguments": {
                "n_iters": None,
                "percentage": None,
                "batch_size": None,
            },
        }

    ######################
    # Tokenization methods
    ######################

    def tokenize(
        self, pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray, **kwargs
    ) -> Dict[str, Any]:

        if self.multicue:
            if (
                self.left_hand_km is None
                or self.right_hand_km is None
                or self.pose_km is None
            ):
                raise ValueError(
                    "Multicue tokenizer not trained yet. Train the model first."
                )
            return self.tokenize_multicue(pose, left_hand, right_hand)
        else:
            if self.all_km is None:
                raise ValueError(
                    "Singlecue tokenizer not trained yet. Train the model first."
                )
        return self.tokenize_singlecue(pose, left_hand, right_hand)

    def tokenize_singlecue(
        self, pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray
    ) -> Dict[str, Any]:

        # Concatenate all the cues into one array
        frames = pose.shape[0]
        data = np.concatenate(
            [
                pose[:, :, :2].reshape(frames, 25 * 2),
                left_hand[:, :, :2].reshape(frames, 21 * 2),
                right_hand[:, :, :2].reshape(frames, 21 * 2),
            ],
            axis=1,
        )

        # Apply the KMeans model to the data (all_km is either Torch or KMeans model)
        predictions = (
            self.all_km.predict(data)
            if hasattr(self.all_km, "predict")
            else self.all_km(data)
        )
        return predictions

    def tokenize_multicue(
        self, pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray
    ) -> Dict[str, Any]:

        frames = pose.shape[0]
        left_hand_predictions = (
            self.left_hand_km.predict(left_hand[:, :, :2].reshape(frames, 21 * 2))
            if hasattr(self.left_hand_km, "predict")
            else self.left_hand_km(left_hand[:, :, :2].reshape(frames, 21 * 2))
        )
        right_hand_predictions = (
            self.right_hand_km.predict(right_hand[:, :, :2].reshape(frames, 21 * 2))
            if hasattr(self.right_hand_km, "predict")
            else self.right_hand_km(right_hand[:, :, :2].reshape(frames, 21 * 2))
        )
        pose_predictions = (
            self.pose_km.predict(pose[:, :, :2].reshape(frames, 25 * 2))
            if hasattr(self.pose_km, "predict")
            else self.pose_km(pose[:, :, :2].reshape(frames, 25 * 2))
        )
        return {
            "left_hand": left_hand_predictions,
            "right_hand": right_hand_predictions,
            "pose": pose_predictions,
        }

    ######################
    ## Training methods
    ######################

    def train(self, X, **kwargs) -> None:
        if self.multicue:
            self.train_multicue(X, **kwargs)
        self.train_singlecue(X, **kwargs)

    @abstractmethod
    def train_singlecue(self, **kwargs) -> None:
        """
        Train the KMeans model for all cues together.
        Every class that inherits from BaseQuantizer should implement this method.
        """
        pass

    @abstractmethod
    def train_multicue(self, **kwargs) -> None:
        """
        Train the KMeans model for each cue separately.
        Every class that inherits from BaseQuantizer should implement this method.
        """
        pass

    ######################
    # IO methods
    ######################

    def load_from_config(self, export_dir: str, model_hash: str) -> None:
        """
        Load the KMeans model and the config file from the export_dir/config_hash directory.
        The config file contains the configuration of the model.

        Args:
        - export_dir: The directory where the model is saved
        - model_hash: The hash of the model configuration
        """

        model_path = os.path.join(export_dir, model_hash)

        # LOAD THE CONFIG FILE
        with open(os.path.join(model_path, "config.json"), "r") as f:
            self.config = json.load(f)

        # LOAD THE KMEANS MODEL
        if self.config["multicue"] and self.config["model_type"] == "kmeans":
            self.left_hand_km = pickle.load(
                open(os.path.join(model_path, "left_hand-km.pkl"), "rb")
            )
            self.right_hand_km = pickle.load(
                open(os.path.join(model_path, "right_hand-km.pkl"), "rb")
            )
            self.pose_km = pickle.load(
                open(os.path.join(model_path, "pose-km.pkl"), "rb")
            )

        elif self.config["multicue"] and self.config["model_type"] != "kmeans":

            # Load the torch models
            self.left_hand_km = torch.load(
                os.path.join(model_path, f"left_hand-{self.config['model_id']}.pt")
            )
            self.right_hand_km = torch.load(
                os.path.join(model_path, f"right_hand-{self.config['model_id']}.pt")
            )
            self.pose_km = torch.load(
                os.path.join(model_path, f"pose-{self.config['model_id']}.pt")
            )

            # Add the model to device
            self.left_hand_km.to(self.device)
            self.right_hand_km.to(self.device)
            self.pose_km.to(self.device)

        elif self.config["model_type"] != "kmeans":
            # Load the torch model
            self.all_km = torch.load(
                os.path.join(model_path, f"{self.config['model_id']}.pt")
            )
            self.all_km.to(self.device)

        else:
            self.all_km = pickle.load(
                open(os.path.join(model_path, "all-km.pkl"), "rb")
            )

        return self

    def save(self, export_dir: str = "models") -> None:
        """
        Export the KMeans model and the config file to the export_dir/config_hash directory.
        The config file contains the configuration of the model.

        File structure:
        ```
        model_id
        |--- {dataset_id}
        |    |--- train.json
        |    |--- val.json
        |    |--- test.json
        |
        |--- {model_id}.pkl  # KMeans model
        |--- {model_id}.json # Config file
        ```
        """
        model_id = self.model_id
        # Export the KMeans model and the config file and the predictions to the export_dir/config_hash
        export_dir = os.path.join(export_dir, model_id)
        os.makedirs(export_dir, exist_ok=True)

        # Export the config file
        with open(os.path.join(export_dir, f"config.json"), "w") as f:
            json.dump(self.config, f)

        # Export the KMeans model(s)
        # TODO: Handle torch models and KMeans models separately
        if self.multicue:
            with open(os.path.join(export_dir, "left_hand-km.pkl"), "wb") as f:
                pickle.dump(self.left_hand_km, f)
            with open(os.path.join(export_dir, "right_hand-km.pkl"), "wb") as f:
                pickle.dump(self.right_hand_km, f)
            with open(os.path.join(export_dir, "pose-km.pkl"), "wb") as f:
                pickle.dump(self.pose_km, f)
        else:
            with open(os.path.join(export_dir, "all-km.pkl"), "wb") as f:
                pickle.dump(self.all_km, f)

        logger.info(f"Model saved to {export_dir}")

    ######################
    # Visualization methods
    #######################

    def _visualize_tokens_for_kmeans(
        self,
        dataset_id: str,
        dataset_type: str,
        dataset_path: str,
        pose_model: str = "openpose",
    ) -> None:

        dataset_path = "data/h5"
        data = self._load_dataset(dataset_id, dataset_type, dataset_path, "all")

        for i, centroid in enumerate(self.all_km.cluster_centers_):

            pose = centroid[: data["pose"].shape[1] * 2]
            left_hand = centroid[
                data["pose"].shape[1] * 2 : data["pose"].shape[1] * 2
                + data["left_hand"].shape[1] * 2
            ]
            right_hand = centroid[
                data["pose"].shape[1] * 2 + data["left_hand"].shape[1] * 2 :
            ]

            pose = pose.reshape(-1, 2)
            left_hand = left_hand.reshape(-1, 2)
            right_hand = right_hand.reshape(-1, 2)

            def viz(fig, frame, connections):
                # get mean of all landmarks and plot in blue
                # TODO: Do not visualize if the landmark is (0,0)
                nonzeros = np.count_nonzero(frame, axis=1)
                # fig.scatter(frame[nonzeros, 0], frame[nonzeros, 1], c='b')
                for connection in connections:
                    fig.plot(
                        [frame[connection[0]][0], frame[connection[1]][0]],
                        [frame[connection[0]][1], frame[connection[1]][1]],
                        "r-",
                    )

            logger.info(centroid[: data["pose"].shape[1] * 2 - 2])
            # visualize all in one plot with subplots
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].set_title("Pose")
            viz(axs[0], pose, POSE_CONNECTION_OPENPOSE)
            axs[1].set_title("Left Hand")
            viz(axs[1], left_hand, HAND_CONNECTIONS)
            axs[2].set_title("Right Hand")
            viz(axs[2], right_hand, HAND_CONNECTIONS)
            plt.show()
