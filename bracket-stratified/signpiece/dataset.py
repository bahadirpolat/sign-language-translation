import os
import h5py
import torch
import evaluate
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any, Optional

from tqdm import tqdm
# from signpiece.transforms import SignTransform, TextTransform

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

# Load the metrics
# bleu = evaluate.load("bleu")
# bleurt_metric = evaluate.load("bleurt")
# rouge = evaluate.load("rouge")
# wer_metric = evaluate.load("wer")

class SignLanguageDataset(Dataset):
    # TODO: Add support for multiple h5 file read
    def __init__(
        self,
        mode: str = "train",
        dataset_id: str = "h2s",
        dataset_path: str = "data/How2Sign",
        **kwargs,
    ) -> None:
        """
        This class is responsible for loading and preprocessing the dataset.
        """
        self.h5_file = h5py.File(os.path.join(dataset_path, f'{dataset_id}.{mode}.h5'), "r")
        self.file_ids = list(self.h5_file.keys())

        self.sign_transform = None if "sign_transform" not in kwargs else kwargs["sign_transform"]
        self.target_transform = None if "target_transform" not in kwargs else kwargs["target_transform"]

        self.percentage = 1.0 if "percentage" not in kwargs else kwargs["percentage"]
        self.mode = mode

    def _get_dataset_from_id(self, dataset_id, mode):
        return h5py.File(os.path.join(self.dataset_path, f'{dataset_id}.{mode}.h5'), "r")

    def __len__(self):
        return len(self.h5_file.keys())

    def __getitem__(self, index):

        left_hand_keypoints = self.h5_file[self.file_ids[index]]["left_hand_keypoints"][:]
        right_hand_keypoints = self.h5_file[self.file_ids[index]]["right_hand_keypoints"][:]
        face_keypoints = self.h5_file[self.file_ids[index]]["face_keypoints"][:]
        pose_keypoints = self.h5_file[self.file_ids[index]]["pose_keypoints"][:]
        text = self.h5_file[self.file_ids[index]]["text"][()].decode("utf-8")

        if self.sign_transform:
            (
                left_hand_keypoints,
                right_hand_keypoints,
                face_keypoints,
                pose_keypoints,
            ) = self.sign_transform(
                left_hand_keypoints,
                right_hand_keypoints,
                face_keypoints,
                pose_keypoints,
            )
        if self.target_transform:
            text = self.target_transform(text)

        return {
            "left_hand": left_hand_keypoints,
            "right_hand": right_hand_keypoints,
            "face": face_keypoints,
            "pose": pose_keypoints,
            "text": text,
            "id": self.file_ids[index],
        }

    def visualize(self, 
                  index: int, 
                  connections: List[Tuple[int, int]],
                  normalize: Optional[List] = None,
                  export: Optional[bool] = None,
                  keypoint_type: Optional[str] = "openpose"
                  ) -> None:
        """
        Visualize the keypoints for a given index in the dataset.

        <HINT>
            - You can use the `mp_drawing` module from the MediaPipe library to draw the connections between the keypoints.
            - You can use the `mp_hands.HAND_CONNECTIONS` and `mp_pose.POSE_CONNECTIONS` to get the connections for hands and pose keypoints respectively.
            - You can use the `plt.scatter` and `plt.plot` functions to visualize the keypoints and connections respectively.
        </HINT>

        <NOTE>
            - The `keypoint_type` parameter can be used to specify the type of keypoints to visualize (e.g.,`"openpose"`, `"mediapipe"`, `"body_135"`).
            - Should discard low confidence points or (0,0) or "undetected" points
        </NOTE>

        Args:
            index (int): Index of the sample to visualize
            connections (List[Tuple[int, int]]): List of connections to draw between keypoints
            normalize (Optional[List]): List of normalization values to apply to the keypoints
            export (Optional[bool]): Whether to export the visualization to a file
        
        Returns:
            None
        """
        # TODO: Implement video visualization logic here
        # TODO: Implement separate visualization 
        #           for face, pose, left hand, and right hand keypoints
        keypoints = self.h5_file[self.file_ids[index]]["keypoints"][:]

        for idx in range(0, keypoints.shape[0]):
            frame = keypoints[idx, :]
            # get mean of all landmarks and plot in blue
            plt.scatter(frame[:, 0], frame[:, 1], c="b")
            for connection in connections:
                plt.plot(
                    [frame[connection[0]][0], frame[connection[1]][0]],
                    [frame[connection[0]][1], frame[connection[1]][1]],
                    "r-",
                )
            # get plt object as an image
            plt.show()

    def for_kmeans(self) -> Dict:

        import random
        random.seed(42)
        dataset = [ ]
        # Samle the dataset
        for file_id in tqdm(self.file_ids, desc=f'Sampling {self.mode} for KMeans'):

            pose_keypoint = self.h5_file[file_id]["pose_keypoints"][:]
            left_hand_keypoint = self.h5_file[file_id]["left_hand_keypoints"][:]
            right_hand_keypoint = self.h5_file[file_id]["right_hand_keypoints"][:]
            face_keypoint = self.h5_file[file_id]["face_keypoints"][:]

            # sample frames from each video
            frames = random.sample(range(pose_keypoint.shape[0]), int(pose_keypoint.shape[0] * self.percentage))
            pose_keypoint = pose_keypoint[frames]
            left_hand_keypoint = left_hand_keypoint[frames]
            right_hand_keypoint = right_hand_keypoint[frames]
            face_keypoint = face_keypoint[frames]

            dataset.append({
                'id': file_id,
                "pose": pose_keypoint,
                "left_hand": left_hand_keypoint,
                "right_hand": right_hand_keypoint,
                "face": face_keypoint,
                'frames': pose_keypoint.shape[0],
                'translation': self.h5_file[file_id]["text"][()].decode("utf-8")
            })
        
        return dataset        

class SignLanguageDatasetForTranslation(SignLanguageDataset):
    def __init__(
        self,
        dataset_id: str,
        sign_tokenizer: Optional[Any] = None,
        text_tokenizer: Optional[Any] = None,
        max_text_length: int = 128,
        max_sign_length: int = 128,
        prompt: str = "Translate American Sign Language to English: ",
        mode: str = "train",
    ) -> None:
        """
        This class is responsible for loading and preprocessing the dataset.

        Args:
        - h5_file: Path to the HDF5 file containing the dataset
        - sign_tokenizer: Tokenizer for the sign language data
        - text_tokenizer: Tokenizer for the text data
        - max_text_length: Maximum length of the text sequence
        - max_sign_length: Maximum length of the sign sequence
        - prompt: Prompt to be used for the text generation task
        """

        super(SignLanguageDatasetForTranslation, self).__init__(dataset_id)

        self.sign_tokenizer = sign_tokenizer
        self.text_tokenizer = text_tokenizer

        self.prompt = prompt

    def to_hf_dataset(self, parquet_file_path: str) -> Any:
        # Implement conversion to HuggingFace dataset logic here
        # Better if you read it from quantized inference file
        # TODO: Read from inference file with quantized model:
        # JSON -> {
        #   "text": "Translate ASL to English: ",
        #   "source_language": "asl", "target_language": "english",
        #   "quantizer": "quantizer_model_id",
        #   "augmentation": "s-f"
        #  }

        train_data_files = parquet_file_path.format("train")
        val_data_files = parquet_file_path.format("val")
        test_data_files = parquet_file_path.format("test")
        
        # Load the dataset
        train_dataset = load_dataset("parquet", data_files=train_data_files, split="train")
        val_dataset = load_dataset("parquet", data_files=val_data_files, split="validation")
        test_dataset = load_dataset("parquet", data_files=test_data_files, split="test")

        train_tokenized = train_dataset.map(self.preprocess_function, batched=True)
        val_tokenized = val_dataset.map(self.preprocess_function, batched=True)
        test_tokenized = test_dataset.map(self.preprocess_function, batched=True)

        return train_tokenized, val_tokenized, test_tokenized

    def preprocess_function(self, examples):
        inputs = [self.prompt + example for example in examples["tokens"]]
        targets = [example for example in examples["text"]]
        model_inputs = self.text_tokenizer(
            inputs,
            text_target=targets,
            max_length=self.max_text_length,
            truncation=True,
        )
        return model_inputs

    def compute_metrics(self, eval_preds):
        # NOTE: This is embedded due to the use of the `tokenizer` variable
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.text_tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )
        labels = np.where(labels != -100, labels, self.text_tokenizer.pad_token_id)

        decoded_labels = self.text_tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # Calculate BLEU, BLEU-1, BLEU-2, BLEU-3, BLEU-4
        bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_scores = bleu_result["bleu"]
        bleu_1, bleu_2, bleu_3, bleu_4 = bleu_result["precisions"]
        # Calculate BLEURT
        bleurt_result = bleurt_metric.compute(
            predictions=decoded_preds, references=[ref[0] for ref in decoded_labels]
        )
        # Calculate ROUGE
        rouge_result = rouge.compute(
            predictions=decoded_preds,
            references=[ref[0] for ref in decoded_labels],
            use_stemmer=True,
        )
        # Calculate WER
        wer_result = wer_metric.compute(
            predictions=decoded_preds, references=[ref[0] for ref in decoded_labels]
        )
        return {
            "bleu": bleu_scores,
            "bleu-1": bleu_1,
            "bleu-2": bleu_2,
            "bleu-3": bleu_3,
            "bleurt": bleurt_result["scores"][0],
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"],
            "rougeLsum": rouge_result["rougeLsum"],
            "wer": wer_result,
        }


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
