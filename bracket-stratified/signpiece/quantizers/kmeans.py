import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import h5py
import pickle
import numpy as np
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
from ..quantizer import BaseQuantizer
from sklearn.cluster import MiniBatchKMeans
from typing import Tuple, List, Dict, Any, Optional
# from signpiece.transforms import SignPieceTransforms
import mediapipe as mp

import logging

logging.basicConfig(level=logging.INFO)
logging.BASIC_FORMAT = "%(asctime)s - %(message)s"

logger = logging.getLogger(name='[KMeansCharTokenizer]')

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

POSE_CONNECTION_OPENPOSE = [
    connection for connection in POSE_CONNECTIONS if connection[0] < 24 and connection[1] < 24
]

# IDEA: What if we calculate the KMeans for each cue separately?
#      This way we can have a more fine-grained representation of the data
#      and we can also have a more interpretable representation of the data
#      since we can see which clusters are more common for each cue

# IDEA: Can we measure distance only with appropriately detected keypoints?
#      The formulation can be \sum_
# IDEA: This is a naive approach, we should consider the temporal window
class KMeansCharTokenizer(BaseQuantizer):
    """
    KMeansCharTokenizer is a class that tokenizes the input data using KMeans clustering.
    This class can be used to tokenize the input data into a sequence of tokens. 

    FIXME: Do not hardcode the number of keypoints in the tokenizer.
    FIXME: Dont hardcode the number of keypoints
    TODO: Export token visualizations as well
    TODO: Add enum for dataset_id and dataset_type
    
    Attributes:
    - multicue: If True, the tokenizer will tokenize the data using multiple KMeans models.
    - num_codebooks: The number of codes for the singlecue tokenizer.
    - left_hand_codebooks: The number of codes for the left hand.
    - right_hand_codebooks: The number of codes for the right hand.
    - pose_codeboo: The number of codes for the pose.
    - temporal_window: The number of frames to consider for the tokenization.
    - normalize: If True, the data will be normalized before tokenization.
    """

    def __init__(self,
                 model_id: str = 'kmeans',
                 multicue: bool = False,
                 num_codebooks: Optional[int] = None,
                 left_hand_codebooks: Optional[int] = None,
                 right_hand_codebooks: Optional[int] = None,
                 pose_codebooks: Optional[int] = None,
                 temporal_window: int = 1,
                 normalizations: List[str] = None,
                 **kwargs
                 ) -> None:
        
        # Init base class with multiple parameters as well as base parameters
        config = {
            'model_id': model_id,
            'multicue': multicue,
            'num_codebooks': num_codebooks,
            'left_hand_codebooks': left_hand_codebooks,
            'right_hand_codebooks': right_hand_codebooks,
            'pose_codebooks': pose_codebooks,
            'temporal_window': temporal_window,
            'normalizations': normalizations
        }
        super(KMeansCharTokenizer, self).__init__(**config)

    ######################
    ## Training methods
    ######################

    def train_singlecue(self, data) -> None:

        print(f"Training KMeans with {len(data)} videos")

        print('Normalizations:', self.normalizations)
        if self.normalizations:
            for norm in self.normalizations:
                cue, norms = list(norm.items())[0]
                print(f"Normalizing {cue} with {norms}")
                for norm in norms:
                    if norm == 'none': continue
                    if norm == 'minmax': data = [self.normalize(d) for d in data]
                    if norm == 'zscore': data = [self.normalize(d) for d in data]
    
        data = [
            np.concatenate([
                d['pose'][:,:,:2].reshape(d['frames'], 25*2), 
                d['left_hand'][:,:,:2].reshape(d['frames'], 21*2), 
                d['right_hand'][:,:,:2].reshape(d['frames'], 21*2)
            ], axis=1 ) for d in tqdm(data)
        ]

        data = np.concatenate(data, axis=0)
        print(f"Training KMeans with {data.shape} frames")
        
        # Train the tokenizer
        self.all_km = KMeans(n_clusters=self.num_codebooks, n_init=42)
        self.all_km.fit(data)

        print('[KM] Trained KMeans with ineartia', self.all_km.inertia_)


    def train_multicue(self, data) -> None:
        
        if not self.left_hand_codebooks or not self.right_hand_codebooks or not self.pose_codebooks:
            raise ValueError("Please specify the number of codes for each cue.")
    
        data = {
            'pose': np.concatenate([d['pose'][:,:,:2].reshape(d['frames'], 25*2) for d in data], axis=0),
            'left_hand': np.concatenate([d['left_hand'][:,:,:2].reshape(d['frames'], 21*2) for d in data], axis=0),
            'right_hand': np.concatenate([d['right_hand'][:,:,:2].reshape(d['frames'], 21*2) for d in data], axis=0)
        }
        
        for d, arr in data.items(): print(f"[{d.upper()}] Training KMeans with {arr.shape} samples")

        # Train the tokenizers
        self.left_hand_km = KMeans(n_clusters=self.left_hand_codebooks, n_init=42)
        self.left_hand_km.fit(data['left_hand'])

        print('[KM] Trained KMeans for left hand with inertia', self.left_hand_km.inertia_)

        self.right_hand_km = KMeans(n_clusters=self.right_hand_codebooks, n_init=42)
        self.right_hand_km.fit(data['right_hand'])

        print('[KM] Trained KMeans for right hand with inertia', self.right_hand_km.inertia_)

        self.pose_km = KMeans(n_clusters=self.pose_codeboo, n_init=42)
        self.pose_km.fit(data['pose'])

        print('[KM] Trained KMeans for pose with inertia', self.pose_km.inertia_)