import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
from fast_pytorch_kmeans import KMeans

class KMeansQuantizer():

    def __init__(self, 
                 codebook_size, 
                 multicue=False,
                 dist_fn='cosine',
                 max_train_instances=None,
                 cues=['pose', 'left_hand', 'right_hand', 'face'],
                 add_factor=10,
                 verbose=False):
        
        self.codebook_size = codebook_size
        self.dist_fn = dist_fn
        self.verbose = verbose

        # Dataset parameters
        self.cues = cues
        self.add_factor = add_factor
        self.multicue = multicue

        self.max_train_instances = max_train_instances

        self.kmeans = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, dataset):

        if not self.multicue:
            # Load the dataset
            X = self.load_singlecue_dataset(dataset)
            X = X * self.add_factor
            # Initialize the model
            self.kmeans = KMeans(
                n_clusters=self.codebook_size, 
                mode='cosine',
                verbose=True
            )
            # Fit the model
            self.kmeans.fit(X)
        else:
            self.kmeans = {}
            for cue in self.cues:
                # Load the dataset
                X = self.load_multicue_dataset(dataset, cue)
                # Initialize the model
                kmeans = KMeans(
                    n_clusters=self.codebook_size, 
                    mode=self.dist_fn,
                    verbose=self.verbose
                )
                # Fit the model
                kmeans.fit(X)
                self.kmeans[cue] = kmeans

    def load_multicue_dataset(self, dataset, cue):
        X = []
        for i in tqdm(range(len(dataset)), desc='Loading train instances for {}'.format(cue)):
            x = dataset[i]
            X.append(x[cue])

        X = torch.cat(X, dim=0)
        if self.max_train_instances is not None:
            # Create random indices
            indices = torch.randperm(X.shape[0])[:self.max_train_instances]
            X = X[indices]
        
        X.to(self.device)

        return X

    def load_singlecue_dataset(self, dataset):

        from signpiece.datasets import POSE_LANDMARKS

        def filter_hands(x):
            # If no pose landmarks are present, skip the instance
            if torch.sum(x[:len(POSE_LANDMARKS)*3]) == 0:
                return
            
            # If both hands are not present, skip the instance
            if torch.sum(x[
                len(POSE_LANDMARKS)*3:
                (21*3*2) + (len(POSE_LANDMARKS)*3) - 1]) == 0:
                return

            return True

        X = []
        for i in tqdm(range(len(dataset)), desc='Loading train instances'):
            _, x, _ = dataset[i]
            # Filter if hands are not present

            # reshape the data from (batch, seq_len, features) to (batch*seq_len, features)
            x = x.reshape(x.shape[0], -1)
            X.extend([
                x[i] for i in range(x.shape[0])
                if filter_hands(x[i])
            ]) # append each sequence as a separate instance
            if len(X) > self.max_train_instances:
                break

        print('Loaded {} instances for training'.format(len(X)))
        print('Each instance has shape: {}'.format(X[0].shape))

        X = torch.stack(X, dim=0)

        if self.max_train_instances is not None:
            indices = torch.randperm(X.shape[0])[:self.max_train_instances]
            X = X[indices]

        print('Loaded {} for training'.format(X))

        X.to(self.device)

        print('Loaded {} instances for training'.format(X.shape))
        return X

    def infer_multicue(self, x):
        if self.kmeans is None: raise ValueError('Model not trained yet. Please call fit() method first.')
        results = {}
        for cue in self.cues:
            results[cue] = self.kmeans[cue].predict(x[cue].reshape(x[cue].shape[0], -1))
        return results

    def quantize(self, x):
        x = x * self.add_factor
        if not self.multicue:
            if self.kmeans is None: raise ValueError('Model not trained yet. Please call fit() method first.')
            return self.kmeans.predict(x).numpy()
        else:
            return self.infer_multicue(x)

    def infer_multicue_dataset(self, dataset, save_dir, mode='train'):
        import h5py

        with h5py.File(save_dir, 'a') as f:
            for file_id, data, sentence in tqdm(dataset, desc=f'Infering {mode} dataset'):
                results = self.infer_multicue(data)
                for cue, data in results.items():                
                    # overwrite the cue data with the cluster labels
                    if file_id not in f.keys():
                        f.create_group(f'{file_id}/features')
                    if cue in f[file_id]['features']:
                        del f[f'{file_id}/features/{cue}']
                    if 'sentence' in f[file_id]:
                        del f[f'{file_id}/sentence']

                    f.create_dataset(f'{file_id}/sentence', data=sentence)
                    f.create_dataset(f'{file_id}/features/{cue}', data=data)

    def infer_singlecue_dataset(self, dataset, save_dir, mode='train'):
        import h5py

        with h5py.File(save_dir, 'a') as f:
            for file_id, data, sentence in tqdm(dataset, desc=f'Infering {mode} dataset'):
                data = self.quantize(data)
                if file_id not in f.keys():
                    f.create_group(f'{file_id}/features')
                if 'features' in f[file_id].keys():
                    del f[f'{file_id}/features']
                if 'sentence' in f[file_id]:
                    del f[f'{file_id}/sentence']

                f.create_dataset(f'{file_id}/sentence', data=sentence)
                f.create_dataset(f'{file_id}/features', data=data)

    def infer_dataset(self, dataset, save_dir, mode='train'):
        if not self.multicue:
            self.infer_singlecue_dataset(dataset, save_dir, mode)
        else:
            self.infer_multicue_dataset(dataset, save_dir, mode)

    def transform(self, X):
        return self.kmeans.predict(X)

    def get_codebook(self):
        return self.kmeans.centroids

    def get_codebook_affine_params(self):
        return None
    
    def load_model(self, model_path):
        if not self.multicue:
            self.kmeans = torch.load(model_path)

    def save_model(self, save_dir):
        if not self.multicue:
            torch.save(self.kmeans, save_dir)
        else:
            for cue, kmeans in self.kmeans.items():
                torch.save(kmeans, os.path.join(save_dir, f'kmeans_{cue}.pt'))

    def get_cluster_distribution(self, labels):
        unique, counts = torch.unique(labels, return_counts=True)
        df_counts = list(zip(unique.cpu().numpy(), counts.cpu().numpy()))
        df = pd.DataFrame(df_counts).rename(columns={0: 'Cluster', 0: 'Count'})
        return df.sort_values('Count', ascending=False).head(20)
    
    def visualize_cluster_centers(self, save_dir):
    
        # Decode all codebook entries and visualize keypoints with mediapipe
        from signpiece.datasets import POSE_LANDMARKS

        POSE_CONNECTIONS = [
            (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (11, 23), (12, 14), (12, 24), (13, 15), 
            (14, 16), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (17, 19),
            (18, 20), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (27, 31), 
            (28, 30), (28, 32), (29, 31), (30, 32),
        ]
        HAND_CONNECTIONS = [
            (0, 1), (0, 5), (0, 17), (1, 2), (2, 3), (3, 4), (5, 6), (5, 9), 
            (6, 7), (7, 8), (9, 10), (9, 13), (10, 11), (11, 12), (13, 14), 
            (13, 17), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)
        ]

        save_path = os.path.join(save_dir, "codebook_reconstruction")
        os.makedirs(save_path, exist_ok=True)

        # delete all files in the directory
        for file in os.listdir(save_path):
            os.remove(os.path.join(save_path, file))

        with torch.no_grad():

            max_display = 200
            for _ in tqdm(range(max_display), desc="Visualizing codebook"):

                idx = np.random.randint(0, self.codebook_size)

                keypoints = self.get_codebook()[idx]

                keypoints = keypoints.reshape(-1, 3)

                # FIXME: This is a hack to make the keypoints visible in the plot
                pose = keypoints[:8, :]
                left_hand = keypoints[8 : 21 + 8, :]
                right_hand = keypoints[21 + 8 :, :]

                # Visualize keypoints with mediapipe connections
                # Create subplot for each body part
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

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
                    axs[0].plot([start[0], end[0]], [start[1], end[1]], "r")

                axs[0].axis("off")
                axs[0].set_title("Pose")

                # Right hand
                for connection in HAND_CONNECTIONS:
                    start = right_hand[connection[0]] * -1
                    end = right_hand[connection[1]] * -1

                    axs[1].plot([start[0], end[0]], [start[1], end[1]], "g")

                axs[1].axis("off")
                axs[1].set_title("Right Hand")

                # Left hand
                for connection in HAND_CONNECTIONS:
                    start = left_hand[connection[0]] * -1
                    end = left_hand[connection[1]] * -1

                    axs[2].plot([start[0], end[0]], [start[1], end[1]], "b")

                axs[2].axis("off")
                axs[2].set_title("Left Hand")

                # Log figure to wandb Image
                # if self.use_wandb:
                #     if self.state["epochs"]+1 % 10 == 0:
                #         self.logger.experiment.log({f"codebook-{idx}": wandb.Image(fig)})

                # Save figure as image
                plt.savefig(os.path.join(save_path, f"codebook_{idx}.png"))

        



        
        

