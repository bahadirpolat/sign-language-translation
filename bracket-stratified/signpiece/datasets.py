import os
import json
import h5py
import re
from responses import start
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset

from transformers import AutoTokenizer
from signpiece.utils.transforms import get_active_frames, normalize_yasl

################################################################# I added library 'random' *alg
import random
import pandas as pd
from torch.utils.data import Dataset
#################################################################

######################################################################
# Bracket Timestamp Scaling for Alignment Task
######################################################################

def scale_bracket_timestamps(text, scale_factor=10):
    """
    Scale timestamps in bracket format by multiplying by scale_factor.
    Input: "[0.5] hello [1.2] world [2.7]"  (original 0.1 precision)
    Output: "[5] hello [12] world [27]"     (scaled ×10 integers)
    
    This makes timestamps easier for the model to learn:
    - Integer tokens are simpler: [25] vs [2.5]
    - Reduced vocabulary: ~600 integer tokens vs infinite decimals
    - Better generalization: clear numeric progression
    """
    if not text or not isinstance(text, str):
        return text
    
    def scale_match(match):
        val = float(match.group(1))
        scaled = int(round(val * scale_factor))
        return f'[{scaled}]'
    
    # Match float timestamps in brackets: [X.X]
    pattern = r'\[(\d+\.\d+)\]'
    return re.sub(pattern, scale_match, text)

######################################################################
POSE_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]

FACE_LANDMARKS = [
    0,
    4,
    13,
    14,
    17,
    33,
    37,
    39,  # Mouth
    46,
    52,
    55,
    61,
    64,
    81,
    82,
    93,  # Eyebrows
    133,
    151,
    152,
    159,
    172,
    178,
    181,  # Eyes
    263,
    269,
    276,
    282,
    285,
    291,
    294,  # Nose
    311,
    323,
    362,
    386,
    397,
    468,
    473,
]
################################################################# I added helper functions for DatasetForPose *alg
def sample_task(task_weights):
    """
    Sample a task based on weights.
    
    Args:
        task_weights: Dict like {'sign2text': 0.90, 'alignment': 0.10}
    
    Returns:
        Task name ('sign2text' or 'alignment')
    """
    tasks = list(task_weights.keys())
    weights = list(task_weights.values())
    return random.choices(tasks, weights=weights)[0]

def is_multi_segment_clip(clip_text_with_timestamps):
    """Check if clip has multiple segments for alignment eligibility"""
    if pd.isna(clip_text_with_timestamps) or not clip_text_with_timestamps:
        return False
    
    import re
    # Count timestamp pairs - more than 1 = multi-segment
    timestamp_pairs = re.findall(r'\d+,\d+,', str(clip_text_with_timestamps))
    return len(timestamp_pairs) > 1
#################################################################
class DatasetForPose(Dataset):

    def __init__(
        self,
        h5_fpath,
        transform="yasl",
        max_instances=None,
        trim_face=False,
        multicue=False,
        use_context=False,
    ):
        self.transform = transform
        self.max_instances = max_instances
        self.h5file = h5py.File(h5_fpath, "r")
        self.multicue = multicue
        self.trim_face = trim_face
        self.use_context = use_context

        self.num_instances = len(list(self.h5file.keys()))

    def __len__(self):
        return self.num_instances if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):

        file_id = list(self.h5file.keys())[idx]
        data = self.h5file[file_id]

        pose_landmarks = data["joints"]["pose"][()]
        face_landmarks = data["joints"]["face"][()]
        left_hand_landmarks = data["joints"]["left_hand"][()]
        right_hand_landmarks = data["joints"]["right_hand"][()]
        sentence = data["sentence"][()].decode("utf-8")
        if self.use_context:
            context = [context.decode("utf-8") for context in data["context"][()]]

        if self.transform:
            if self.transform == "yasl":
                pose_landmarks = normalize_yasl(pose_landmarks)
                face_landmarks = normalize_yasl(face_landmarks)
                left_hand_landmarks = normalize_yasl(left_hand_landmarks)
                right_hand_landmarks = normalize_yasl(right_hand_landmarks)
            else:
                raise NotImplementedError(
                    f"{self.transform} normalization is not implemented yet"
                )

        # Select only the keypoints that are needed
        pose_landmarks = pose_landmarks[:, POSE_LANDMARKS, :]
        face_landmarks = (
            face_landmarks
            if not self.trim_face
            else face_landmarks[:, FACE_LANDMARKS, :]
        )

        # Remove last 1 channel (visibility)
        pose_landmarks = pose_landmarks[:, :, :3]
        face_landmarks = face_landmarks[:, :, :3]
        left_hand_landmarks = left_hand_landmarks[:, :, :3]
        right_hand_landmarks = right_hand_landmarks[:, :, :3]

        # Convert keypoints to tensor
        pose_landmarks = torch.tensor(pose_landmarks, dtype=torch.float)
        face_landmarks = torch.tensor(face_landmarks, dtype=torch.float)
        left_hand_landmarks = torch.tensor(left_hand_landmarks, dtype=torch.float)
        right_hand_landmarks = torch.tensor(right_hand_landmarks, dtype=torch.float)

        # Concatenate all keypoints
        keypoints = torch.cat(
            (
                pose_landmarks,
                left_hand_landmarks,
                right_hand_landmarks,
                face_landmarks,
            ),
            dim=1,
        )
        # Reduce the keypoints (T, N, C) -> (T, N*C)
        keypoints = keypoints.view(keypoints.size(0), -1)
        # Check if keypoints are in the correct shape
        # assert keypoints.shape[-1] == 255, "Key points are not in the correct shape"

        if self.use_context:
            return file_id, keypoints, sentence, context

        return file_id, keypoints, sentence


class DatasetForTokenization(Dataset):

    def __init__(
        self,
        h5_fpath,
        dataset_id,
        mode="train",
        input_type="char",
        skip_frames=True,
        max_token_length=128,
        max_sequence_length=250,
        tokenizer="google-t5/t5-small",
        prompt="translate ASL to English: ",
        multicue=False,
        use_bpe=False,
    ):

        self.mode = mode
        h5_fpath = os.path.join(
            h5_fpath,
            ".".join([dataset_id, ("char" if not use_bpe else "bpe"), mode] + ["h5"]),
        )
        self.h5file = h5py.File(h5_fpath, "r")

        self.input_type = input_type
        self.multicue = multicue
        self.prompt = prompt

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.max_sequence_length = max_sequence_length
        self.skip_frames = skip_frames

        if self.input_type == "quantized" and skip_frames:
            raise ValueError("skip_frames should be False for `quantized` input")

    def __len__(self):

        return len(list(self.h5file.keys()))

    def __getitem__(self, idx):

        if self.multicue:

            file_id = list(self.h5file.keys())[idx]
            pose_chars = self.h5file[file_id]["features"]["pose"][()]
            left_hand_chars = self.h5file[file_id]["features"]["left_hand"][()]
            right_hand_chars = self.h5file[file_id]["features"]["right_hand"][()]
            face_chars = self.h5file[file_id]["features"]["face"][()]
            sentence = self.h5file[file_id]["sentence"][()].decode("utf-8")

            return {
                "file_id": file_id,
                "pose": pose_chars,
                "left_hand": left_hand_chars,
                "right_hand": right_hand_chars,
                "face": face_chars,
                "sentence": sentence,
            }

        else:

            file_id = list(self.h5file.keys())[idx]
            features = self.h5file[file_id]["features"][()]
            sentence = self.h5file[file_id]["sentence"][()].decode("utf-8")

            return {
                "file_id": file_id,
                "features": features,
                "sentence": sentence,
            }


class DatasetForQuantization(DatasetForPose):

    def __init__(
        self,
        h5_fpath,
        dataset_id,
        mode="train",
        transform="yasl",
        skip_frames=False,
        max_sequence_length=None,
        multicue=False,
        max_instances=None,
        use_face=False,
        trim_face=False,
    ):

        self.mode = mode
        self.use_face = use_face
        self.multicue = multicue
        self.max_instances = max_instances
        h5_fpath = os.path.join(h5_fpath, ".".join([dataset_id, "pose", mode] + ["h5"]))

        self.max_sequence_length = max_sequence_length
        self.skip_frames = skip_frames

        super().__init__(h5_fpath, transform, max_instances, trim_face, multicue)

    def __len__(self):
        return super().__len__() if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):

        if self.multicue:
            file_id, data, sentence = super().__getitem__(idx)
            pose_landmarks = data["pose"]
            left_hand_landmarks = data["left_hand"]
            right_hand_landmarks = data["right_hand"]
            face_landmarks = data["face"]

            if self.skip_frames:
                pose_landmarks = pose_landmarks[::2]
                left_hand_landmarks = left_hand_landmarks[::2]
                right_hand_landmarks = right_hand_landmarks[::2]
                face_landmarks = face_landmarks[::2]

            if self.max_sequence_length:
                pose_landmarks = pose_landmarks[: self.max_sequence_length]
                left_hand_landmarks = left_hand_landmarks[: self.max_sequence_length]
                right_hand_landmarks = right_hand_landmarks[: self.max_sequence_length]
                face_landmarks = face_landmarks[: self.max_sequence_length]

            keypoints = {
                "pose": pose_landmarks,
                "left_hand": left_hand_landmarks,
                "right_hand": right_hand_landmarks,
                "face": face_landmarks,
            }
        else:
            file_id, keypoints, sentence = DatasetForPose.__getitem__(self, idx)

            if self.skip_frames:
                keypoints = keypoints[::2]
            if self.max_sequence_length:
                keypoints = keypoints[: self.max_sequence_length]

        # if not self.use_face:
        #     return file_id, keypoints[:,:-len(FACE_LANDMARKS)*3], sentence

        return file_id, keypoints, sentence


class DatasetForQuantizationTraining(DatasetForPose):

    def __init__(
        self,
        h5_fpath,
        dataset_id,
        mode="train",
        transform="yasl",
        max_instances=None,
        use_face=False,
    ):
        h5_fpath = os.path.join(h5_fpath, ".".join([dataset_id, "pose", mode] + ["h5"]))
        super().__init__(h5_fpath, transform)

        self.frame_counts = None
        self.total_frames = None
        self.max_instances = max_instances
        self.use_face = use_face
        self._count_frame_sum()

    def _count_frame_sum(self):
        self.frame_counts = []
        for idx in tqdm(range(super().__len__()), desc="Counting frames"):
            self.frame_counts.extend(
                [
                    (idx, frame_idx)
                    for frame_idx in range(super().__getitem__(idx)[1].shape[0])
                ]
            )
            if self.max_instances and len(self.frame_counts) > self.max_instances:
                print("Reached max instances", self.max_instances)
                break

        self.total_frames = len(self.frame_counts)

        if self.max_instances:
            self.total_frames = min(self.total_frames, self.max_instances)

        # Shuffle the frame counts
        self.random_indices = np.random.permutation(self.total_frames)
        self.frame_counts = [self.frame_counts[idx] for idx in self.random_indices]

        self.frame_counts = self.frame_counts[: self.total_frames]

    def __len__(self):
        return self.total_frames if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):
        # frame_counts is a list of file indices repeated for each frame
        # Get the file index AND get frame index by subtracting the index from the last index of the frame_counts
        sentence_idx, frame_idx = self.frame_counts[idx]
        keypoints = super().__getitem__(sentence_idx)[1]
        # EXPAND NORMALIZED KEYPOINTS TO -10 to 10
        keypoints = keypoints * 10
        if not self.use_face:
            return keypoints[frame_idx, : -len(FACE_LANDMARKS) * 3]
        return keypoints[frame_idx]


class DatasetForSLT(Dataset):

    def __init__(
        self,
        h5_fpath,
        dataset_id,
        mode="train",
        skip_frames=True,
        max_token_length=128,
        max_sequence_length=250,
        tokenizer="google-t5/t5-small",
        source_lang="asl",
        target_lang="en",
        num_codes=512,
        multicue=False,
        use_bpe=False,
        use_hf=False,
        objective="sign2text",
        max_instances=None,
        ########################################################## New parameter was added for alg *alg
        task_weights=None,
        ##########################################################
    ):

        # Load the dataset
        self.mode = mode
        h5_fpath = os.path.join(
            h5_fpath,
            ".".join([dataset_id, ("char" if not use_bpe else "bpe"), mode] + ["h5"]),
        )
        self.h5file = h5py.File(h5_fpath, "r")

        # Parameters for signpiece
        self.multicue = multicue
        self.use_bpe = use_bpe
        self.prompt = (
            f"translate {source_lang.upper()} to {target_lang.upper()}: "
            if objective == "sign2text"
            else f"translate {target_lang.upper()} to {source_lang.upper()}: "
        )

        # Huggingface tokenizer for the prompt
        self.use_hf = use_hf
        self.num_codes = num_codes
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.objective = objective
        self.update_tokenizer(tokenizer)

        # Parameters for the dataset (YASL)
        self.max_token_length = max_token_length
        self.max_sequence_length = max_sequence_length
        self.skip_frames = skip_frames
        self.max_instances = max_instances

        # reverse if objective is text2sign
        if self.objective == "text2sign":
            self.max_sequence_length, self.max_token_length = (
                self.max_token_length,
                self.max_sequence_length,
            )

        if self.multicue and self.use_hf and not skip_frames:
            raise ValueError(
                "skip_frames should be True for multicue input, else results in high input size"
            )
        ############################################################## Storing task weights *alg
        if task_weights is None:
            # Default: no alignment task
            task_weights = {'sign2text': 1.0, 'alignment': 0.0}
        self.task_weights = task_weights
        ##############################################################

    def __len__(self):
        return (
            len(list(self.h5file.keys()))
            if self.max_instances is None
            else self.max_instances
        )

    def update_tokenizer(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # Add special tokens for the cues
        self.tokenizer.add_tokens(
            [f"<{self.source_lang}-{idx}>" for idx in range(1, self.num_codes + 1)]
        )

    def _process_multicue(self, file_id, max_length=None):

        sign_tokens = {
            "pose": torch.tensor(self.h5file[file_id]["features"]["pose"][()]),
            "left_hand": torch.tensor(
                self.h5file[file_id]["features"]["left_hand"][()]
            ),
            "right_hand": torch.tensor(
                self.h5file[file_id]["features"]["right_hand"][()]
            ),
            "face": torch.tensor(self.h5file[file_id]["features"]["face"][()]),
        }

        if self.skip_frames:
            for key in ["pose", "left_hand", "right_hand", "face"]:
                sign_tokens[key] = sign_tokens[key][::2]
        if self.max_sequence_length:
            for key in ["pose", "left_hand", "right_hand", "face"]:
                sign_tokens[key] = sign_tokens[key][
                    : max_length if max_length else self.max_sequence_length
                ]

        # Add one more dimension to the sign_tokens
        for key in ["pose", "left_hand", "right_hand", "face"]:
            sign_tokens[key] = sign_tokens[key].unsqueeze(0)

        return sign_tokens

    def _process_singlecue(self, file_id, max_length=None):

        sign_tokens = self.h5file[file_id]["features"][()]

        if self.skip_frames:
            sign_tokens = sign_tokens[::2]
        if self.max_sequence_length:
            sign_tokens = sign_tokens[
                : max_length if max_length else self.max_sequence_length
            ]

        sign_tokens = torch.tensor(sign_tokens)

        return sign_tokens

    def __getitem__(self, idx):
        # Get the sign_tokens and the sentence,

        file_id = list(self.h5file.keys())[idx]
        sentence = self.h5file[file_id]["sentence"][()].decode("utf-8")

        input_ids = self.tokenizer(
            self.prompt,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # Tokenize the sentence
        decoder_input_ids = self.tokenizer(
            sentence,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        if self.multicue:
            sign_tokens = self._process_multicue(
                file_id, max_length=self.max_sequence_length - input_ids.size(1)
            )
        else:
            sign_tokens = self._process_singlecue(
                file_id, max_length=self.max_sequence_length - input_ids.size(1)
            )

        # Add attention mask
        attention_mask = torch.ones(
            input_ids.size(1)
            + (
                sign_tokens.size(0)
                if not self.multicue
                else sign_tokens["pose"].size(1)
            ),
            dtype=int,
        )

        if self.use_hf:
            sign_tokens += self.tokenizer.vocab_size
            if self.objective == "sign2text":
                return {
                    "file_id": file_id,
                    "input_ids": torch.cat([input_ids, sign_tokens[None, ...]], dim=1),
                    "sentence": sentence,
                    "labels": decoder_input_ids,
                    "attention_mask": attention_mask,
                }
            else:
                return {
                    "file_id": file_id,
                    "input_ids": torch.cat([input_ids, decoder_input_ids], dim=1),
                    "sentence": sentence,
                    "labels": sign_tokens[None, ...],
                    "attention_mask": torch.ones(
                        input_ids.size(1) + decoder_input_ids.size(1), dtype=int
                    ),
                }

        return {
            "file_id": file_id,
            "sign_ids": sign_tokens,
            "input_ids": input_ids,
            "sentence": sentence,
            "labels": decoder_input_ids,
            "attention_mask": attention_mask,
        }


class MultiTaskDatasetForSLT(Dataset):
    """
    Wrapper class for multiple datasets
    """

    def __init__(self, datasets, shuffle=True):
        self.datasets = datasets
        self.shuffle = shuffle

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        # Add consistent shuffling
        if self.shuffle:
            idx = torch.randperm(len(self))[idx]

        if idx < len(self.datasets[0]):
            return self.datasets[0][idx]
        elif idx < len(self.datasets[0]) + len(self.datasets[1]):
            return self.datasets[1][idx - len(self.datasets[0])]
        else:
            return self.datasets[2][idx - len(self.datasets[0]) - len(self.datasets[1])]


class HFDatasetForSLT(Dataset):

    def __init__(
        self,
        h5_fpath,
        dataset_id,
        mode="train",
        objective="sign2text",
        max_token_length=128,
        max_sign_length=250,
        tokenizer_name="google-t5/t5-small",
        source_lang="asl",
        target_lang="en",
        num_codes=512,
        annotation_file=None,
        max_instances=None,
        skip_frames=True,
        random_drop=False,
        random_switch=False,
        duplicate_frames=False,
        model_type="t5",
        inference_mode=False,
        use_linear=False,
        use_bpe=False,
        use_char=True,
        use_context=False,
        trim_face=False,
        seed=42,
        random_drop_rate=0.1,
        use_paraphrase=False,
        ################################################# Change according to the dataset type
        dataset_type="main",  # main or long 
        #################################################
        #################################################
        task_weights=None, # <--- ADD THIS LINE *alg
        #################################################
    ):
        # Load the dataset
        self.mode = mode
        self.dataset_type = dataset_type
        self.dataset_id = dataset_id
        self.tokenizer_name = tokenizer_name
        self.model_type = model_type
        self.inference_mode = inference_mode
        self.trim_face = trim_face
        self.use_paraphrase = use_paraphrase

        self.h5_fpath = h5_fpath
        self.seed = seed
        self.random_drop_rate = random_drop_rate

        # Parameters for signpiece
        self.use_char = use_char
        self.use_bpe = use_bpe
        self.use_linear = use_linear
        self.use_context = use_context


        if self.use_linear:
            self.input_type = "pose"
        if self.use_char:
            self.input_type = "char"
        if self.use_bpe:
            self.input_type = "bpe"
        ############################################################### Saving the weights *alg
        if task_weights is None:
            # Default: no alignment task
            task_weights = {'sign2text': 1.0, 'alignment': 0.0}
        self.task_weights = task_weights
        ###############################################################
        # Read from the annotation file and create the idx2instance
        self.annotation_file = annotation_file
        if self.annotation_file:

            # INFO DEPRECATED: Use import_instances_from_csv instead
            def import_instances_from_json(
                annotation_file, use_paraphrase=False, max_instances=None
            ):

                keys = self.h5file.keys()
                print(f"Loaded keys from {h5_fpath}")

                import pandas as pd

                # Read the annotation json file
                annotation_df = pd.read_json(annotation_file, lines=True)
                annotation = pd.read_json(annotation_file, lines=True)

                if use_paraphrase and (mode == "train" or mode == "dev"):
                    print("Instance size before paraphrases:", len(annotation))
                    annotation = annotation.explode("paraphrases")
                    annotation = annotation[annotation["paraphrases"].notnull()]
                    annotation["text"] = annotation["paraphrases"]
                    print("Instance size after paraphrases:", len(annotation))

                annotation = annotation.to_dict(orient="records")

                print(f"Loaded {len(annotation)} instances from the annotation file")
                # Create the idx2instance dictionary
                idx = 0
                idx2instance = {}
                for instance in tqdm(annotation):

                    if max_instances and idx >= max_instances:
                        break

                    # Check if the instance is in the dataset
                    if instance["clip_id"] in keys:
                        idx2instance[idx] = instance
                        idx += 1

                return annotation, idx2instance

            def import_instances_from_csv(
                annotation_file, use_paraphrase=False, max_instances=None
            ):
                import pandas as pd

                # Read the annotation json file
                annotation = pd.read_csv(annotation_file)

                if "paraphrases" in annotation.columns and mode == "train" and use_paraphrase:
                    from ast import literal_eval
                    annotation["paraphrases"] = annotation["paraphrases"].apply(
                        literal_eval
                    )
                    print("Instance size before paraphrases:", len(annotation))
                    annotation = annotation.explode("paraphrases")
                    annotation = annotation[annotation["paraphrases"].notnull()]
                    annotation["text"] = annotation["paraphrases"]
                    # Shuffle dataset
                    annotation = annotation.sample(
                        frac=1, random_state=seed
                    ).reset_index(drop=True)
                    print("Instance size after paraphrases:", len(annotation))

                # check has shard_id
                annotation["has_shard"] = annotation["shard_id"].apply(
                    lambda _id: (
                        True
                        if os.path.exists(
                            os.path.join(
                                h5_fpath,
                                ".".join(
                                    [self.dataset_id, self.dataset_type, str(_id)]
                                    + ["h5"]
                                ),
                            )
                        )
                        else False
                    )
                )
                print("Before filtering non existing shards:", len(annotation))

                annotation = annotation[annotation.has_shard == True]

                print("After filtering non existing shards:", len(annotation))

                # Sort by shard_id
                annotation.sort_values(by="shard_id", inplace=True)

                annotation = annotation.to_dict(orient="records")

                print(f"File has {len(annotation)} instances from the annotation file")
                # Create the idx2instance dictionary
                idx = 0
                idx2instance = {}
                for instance in tqdm(annotation):
                    # Check if num of instances is reached
                    if max_instances and idx >= max_instances:
                        break
                    # Check if the instance is in the dataset
                    idx2instance[idx] = instance
                    idx += 1

                return annotation, idx2instance

            self.annotation, self.idx2instance = import_instances_from_csv(
                annotation_file, use_paraphrase=use_paraphrase, max_instances=max_instances
            )

        self.shard_id = self.annotation[0]["shard_id"]
        h5_fpath = os.path.join(
            h5_fpath,
            ".".join([self.dataset_id, self.dataset_type, str(self.shard_id)] + ["h5"]),
        )

        print(f"Loading dataset from {h5_fpath}")
        self.h5file = h5py.File(h5_fpath, "r")

        print(f"Loaded {len(self.idx2instance)} instances from the annotation file")

        # Huggingface tokenizer for the prompt
        self.num_codes = num_codes
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.prompt = self.generate_prompt(
            objective, source_lang, target_lang, use_context
        )
        self.context_prefix = (
            "\nContext:\n" if source_lang == "English" else "\nBağlam:\n"
        )

        self.objective = objective
        self.source_lang_code = "".join([word[0] for word in source_lang.split(" ")])

        self.tokenizer_name = tokenizer_name
        self.update_tokenizer(tokenizer_name)

        # Parameters for the dataset (YASL)
        self.max_token_length = max_token_length
        self.max_sign_length = max_sign_length
        self.max_instances = max_instances

        # Data augmentation
        self.skip_frames = skip_frames
        self.random_drop = random_drop
        self.random_drop_rate = random_drop_rate
        self.random_switch = random_switch
        self.duplicate_frames = duplicate_frames

    def __len__(self):
        return len(self.idx2instance.keys())

    def _get_paraphrase_by_idx(self, clip_id):
        # # group annotations by clip_id and get paraphrase
        # if self.use_paraphrase:
        #     # return self.annotation[clip_id]["paraphrases"]
        #     return self.annotation[self.annotation["clip_id"] == clip_id]["paraphrases"].values[0]
        return

    def update_tokenizer(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if "gpt" in tokenizer_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"

        # Add bos token to the tokenizer
        # Add special tokens for the cues
        self.tokenizer.add_tokens(
            [f"<{self.source_lang_code}-{idx}>" for idx in range(self.num_codes)]
        )

    def generate_prompt(self, objective, source_lang, target_lang, use_context=False):
        """
        Get the prompt for the objective. The prompt is used to generate the input sequence for the model

        Args:
            objective (str): Objective of the model
            source_lang (str): Source language
            target_lang (str): Target language
            use_context (bool): Use context for the translation

        Returns:
            prompt (str): Prompt for the objective
        """

        print("Generating Prompt for the Objective")
        print(
            f"Objective: {objective}, Source: {source_lang}, Target: {target_lang}, Context: {use_context}"
        )

        # Sign2Text Translation (English to ASL)
        if objective == "sign2text" and target_lang == "English" and use_context:
            return (
                f"Translate {source_lang} to {target_lang} with the following context: "
            )
        elif objective == "sign2text" and target_lang == "English" and not use_context:
            return f"Translate {source_lang} to {target_lang}: "

        # Text2Sign Translation (ASL to English)
        elif objective == "text2sign" and target_lang == "English" and use_context:
            return (
                f"Translate {target_lang} to {source_lang} with the following context: "
            )
        elif objective == "text2sign" and target_lang == "English" and not use_context:
            return f"Translate {target_lang} to {source_lang}: "

        # Sign2Sign Translation (ASL to ASL)
        elif objective == "sign2sign" and target_lang == "English" and use_context:
            return f"Complete the sign sequence for the translation in {target_lang} with the following context: "
        elif objective == "sign2sign" and target_lang == "English" and not use_context:
            return f"Complete the sign sequence for the translation in {target_lang}: "

        # Sign2Text Translation (TID to Turkish)
        if objective == "sign2text" and target_lang == "Türkçe" and use_context:
            return f"{source_lang}'den {target_lang}'ye aşağıdaki bağlamı kullanarak çevir: "
        elif objective == "sign2text" and target_lang == "Türkçe" and not use_context:
            return f"{source_lang}'den {target_lang}'ye çevir: "

        # Text2Sign Translation (Turkish to TID)
        elif objective == "text2sign" and target_lang == "Türkçe" and use_context:
            return f"{target_lang}'den {source_lang}'ye aşağıdaki bağlamı kullanarak çevir: "
        elif objective == "text2sign" and target_lang == "Türkçe" and not use_context:
            return f"{target_lang}'den {source_lang}'ye çevir: "

        # Sign2Sign Translation (TID to TID)
        elif objective == "sign2sign" and target_lang == "Türkçe" and use_context:
            return f"{target_lang} cümlesini aşağıdaki bağlamı kullanarak tamamla: "
        elif objective == "sign2sign" and target_lang == "Türkçe" and not use_context:
            return f"{target_lang} cümlesini tamamla: "

        else:
            raise ValueError(f"Objective {objective} is not supported")
    #################################################################################################################
    def get_discrete(self, sign_tokens, translation, context=None, task_token='<slt>', prompt_text=""): # Task tokens are added *alg
    #################################################################################################################
        match self.objective:
            # SIGN LANGUAGE TRANSLATION
            case "sign2text":
                ################################################################################### Commented, *alg
                """ 
                input_sequence = (
                    #################################################################
                    #"[S2T] " Commented for alg task  + f"{task_token} " added *alg
                    #################################################################
                    f"{task_token} "
                    + self.prompt  # Start of the prompt
                    + (
                        "" if not context else self.context_prefix + "\n".join(context)
                    )  # Context
                    + " "
                    + "".join(
                        [f"<{self.source_lang_code}-{idx}>" for idx in sign_tokens]
                    )  # Given sign sequence
                )
                """
                input_parts = [f"{task_token} ", self.prompt]
            
                if context:
                    input_parts.append(self.context_prefix + "\n".join(context) + " ")
                
                # Add clip_text for alignment task
                if prompt_text:
                    input_parts.append(prompt_text + " ")
                
                # Add sign tokens
                input_parts.append("".join(
                    [f"<{self.source_lang_code}-{idx}>" for idx in sign_tokens]
                ))
                
                input_sequence = "".join(input_parts)

                ###################################################################################
                target_sequence = translation

            # SIGN LANGUAGE PRODUCTION
            case "text2sign":
                input_sequence = (
                    #################################################################
                    #"[T2S] " Commented for alg task  + f"{task_token} " added *alg
                    #################################################################
                    f"{task_token} "
                    + self.prompt  # Start of the prompt
                    + (
                        "" if not context else self.context_prefix + "\n".join(context)
                    )  # Context
                    + " "
                    + translation  # Given translation
                )
                target_sequence = "".join(
                    [f"<{self.source_lang_code}-{idx}>" for idx in sign_tokens]
                )

            # SIGN LANGUAGE COMPLETION
            case "sign2sign":
                random_start = torch.randint(0, max(len(sign_tokens), 1), (1,))
                input_sequence = (
                    #################################################################
                    #"[S2S] " Commented for alg task  + f"{task_token} " added *alg
                    #################################################################
                    f"{task_token} "
                    + self.prompt  # Start of the prompt
                    + (
                        "" if not context else self.context_prefix + "\n".join(context)
                    )  # Context
                    + " "
                    + "".join(
                        [
                            f"<{self.source_lang_code}-{idx}>"
                            for idx in sign_tokens[:random_start]
                        ]
                    )  # Given sign sequence
                )
                target_sequence = "".join(
                    [
                        f"<{self.source_lang_code}-{idx}>"
                        for idx in sign_tokens[random_start:]
                    ]
                )  # Target sign sequence

            case _:
                raise ValueError(f"Objective {self.objective} is not supported")

        return input_sequence, target_sequence

    def get_continuous(self, sign_tokens, translation, context=None, task_token='<slt>', prompt_text=""): # Task token was added *alg
#################################################################################################
        match self.objective:
            # SIGN LANGUAGE TRANSLATION
            case "sign2text":

                """ This part is commented, changed to its new version for random alignment *alg
                
                input_sequence = self.prompt + (
                    "" if not context else self.context_prefix + "\n\n".join(context)
                )
                """
                ############################################################################## Commented, *alg
                """ 
                input_sequence = f"{task_token} " + self.prompt + (
                    "" if not context else self.context_prefix + "\n\n".join(context)
                )
                """
                # Build input sequence with task token
                input_parts = [f"{task_token} ", self.prompt]
                
                # Add context if available
                if context:
                    input_parts.append(self.context_prefix + "\n\n".join(context))
                
                # For alignment task, add clip_text to prompt
                if prompt_text:
                    input_parts.append(" ")
                    input_parts.append(prompt_text)
                
                input_sequence = "".join(input_parts)

                ############################################################################## *alg

                target_sequence = translation

            # SIGN LANGUAGE PRODUCTION
            case "text2sign":
                raise ValueError(
                    "Continuous input is not supported for text2sign objective"
                )

            # SIGN LANGUAGE COMPLETION
            case "sign2sign":
                raise ValueError(
                    "Continuous input is not supported for sign2sign objective"
                )

            case _:
                raise ValueError(f"Objective {self.objective} is not supported")

        # Tokenize the input and target sequences
        model_inputs = self.tokenizer(
            input_sequence,
            text_target=target_sequence,
            max_length=self.max_token_length,
            padding="longest",
            truncation=True,
        )

        text_inputs = model_inputs.input_ids
        text_attention = model_inputs.attention_mask
        # context = context.input_ids
        labels = model_inputs.labels

        # add <EOS> token to labels
        if self.tokenizer_name == "boun-tabi-LMG/TURNA":
            labels = labels + [self.tokenizer.eos_token_id]

        sign_attention = torch.ones(sign_tokens.size(0), dtype=int)

        return (text_inputs, text_attention), (sign_tokens, sign_attention), labels

    def get_pose(self, file_id):
        """
        Get the pose landmarks for the given index

        Args:
            index (int): Index of the dataset

        Returns:
            pose_landmarks (torch.Tensor): Pose landmarks
        
        ################################################################ Error handling for missing data !!!!!! - fixed
        ########################################################## Original code :
        data = self.h5file[file_id]

        pose_landmarks = data["joints"]["pose"][()]
        face_landmarks = data["joints"]["face"][()]
        left_hand_landmarks = data["joints"]["left_hand"][()]
        right_hand_landmarks = data["joints"]["right_hand"][()]
        ################################################################ Error handling for missing data !!!!!!

        pose_landmarks = normalize_yasl(pose_landmarks)
        face_landmarks = normalize_yasl(face_landmarks)
        left_hand_landmarks = normalize_yasl(left_hand_landmarks)
        right_hand_landmarks = normalize_yasl(right_hand_landmarks)
        """
        ####################################################################### *alg
        # Check if file_id contains multiple clips (comma-separated)
        clip_ids = [cid.strip() for cid in file_id.split(',')]
        
        all_pose_landmarks = []
        all_face_landmarks = []
        all_left_hand_landmarks = []
        all_right_hand_landmarks = []
        
        for clip_id in clip_ids:
            try:
                data = self.h5file[clip_id]
                
                pose_landmarks = data["joints"]["pose"][()]
                face_landmarks = data["joints"]["face"][()]
                left_hand_landmarks = data["joints"]["left_hand"][()]
                right_hand_landmarks = data["joints"]["right_hand"][()]
                
                # Normalize each clip
                pose_landmarks = normalize_yasl(pose_landmarks)
                face_landmarks = normalize_yasl(face_landmarks)
                left_hand_landmarks = normalize_yasl(left_hand_landmarks)
                right_hand_landmarks = normalize_yasl(right_hand_landmarks)
                
                all_pose_landmarks.append(pose_landmarks)
                all_face_landmarks.append(face_landmarks)
                all_left_hand_landmarks.append(left_hand_landmarks)
                all_right_hand_landmarks.append(right_hand_landmarks)
            except KeyError:
                print(f"Warning: clip_id '{clip_id}' not found in H5 file, skipping this clip...")
                continue
        
        if not all_pose_landmarks:
            raise ValueError(f"No valid clips found for file_id: {file_id}")
        
        # Concatenate all clips along time dimension (axis 0)
        pose_landmarks = np.concatenate(all_pose_landmarks, axis=0)
        face_landmarks = np.concatenate(all_face_landmarks, axis=0)
        left_hand_landmarks = np.concatenate(all_left_hand_landmarks, axis=0)
        right_hand_landmarks = np.concatenate(all_right_hand_landmarks, axis=0)
        ####################################################################################### *alg
        
        start, end = get_active_frames(pose_landmarks)
        
        # If active frames are found, select only the active frames
        pose_landmarks = pose_landmarks[start:end + 1, :, :]
        face_landmarks = face_landmarks[start:end + 1, :, :]
        left_hand_landmarks = left_hand_landmarks[start:end + 1, :, :]
        right_hand_landmarks = right_hand_landmarks[start:end + 1, :, :]
    

        # Select only the keypoints that are needed
        pose_landmarks = pose_landmarks[:, POSE_LANDMARKS, :]
        face_landmarks = (
            face_landmarks
            if not self.trim_face
            else face_landmarks[:, FACE_LANDMARKS, :]
        )

        # Remove last 1 channel (visibility)
        pose_landmarks = pose_landmarks[:, :, :3]
        face_landmarks = face_landmarks[:, :, :3]
        left_hand_landmarks = left_hand_landmarks[:, :, :3]
        right_hand_landmarks = right_hand_landmarks[:, :, :3]

        # Convert keypoints to tensor
        pose_landmarks = torch.tensor(pose_landmarks, dtype=torch.float)
        face_landmarks = torch.tensor(face_landmarks, dtype=torch.float)
        left_hand_landmarks = torch.tensor(left_hand_landmarks, dtype=torch.float)
        right_hand_landmarks = torch.tensor(right_hand_landmarks, dtype=torch.float)

        # Concatenate all keypoints
        keypoints = torch.cat(
            (
                pose_landmarks,
                left_hand_landmarks,
                right_hand_landmarks,
                face_landmarks,
            ),
            dim=1,
        )
        # Reduce the keypoints (T, N, C) -> (T, N*C)
        keypoints = keypoints.view(keypoints.size(0), -1)
        # Check if keypoints are in the correct shape
        # assert keypoints.shape[-1] == 255, "Key points are not in the correct shape"

        # If nan values are present, replace them with 0
        keypoints.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        return keypoints

    def format_sign_tokens(self, sign_tokens):
        """
        Format the sign tokens based on the data augmentation techniques

        Args:
            sign_tokens (list): List of sign tokens

        Returns:
            sign_tokens (list): Formatted sign tokens
        """

        if self.skip_frames:
            sign_tokens = sign_tokens[::2]
        if self.random_drop:
            random_frames = torch.randint(0, len(sign_tokens), (len(sign_tokens) - 5,))
            sign_tokens = sign_tokens[random_frames]
        if self.random_switch:
            # switch random two frames
            random_frames = torch.randint(0, len(sign_tokens), (2,))
            sign_tokens[random_frames[0]], sign_tokens[random_frames[1]] = (
                sign_tokens[random_frames[1]],
                sign_tokens[random_frames[0]],
            )
        if self.duplicate_frames:
            tokens = []
            for token in sign_tokens:
                tokens.extend([token] * 2)
            sign_tokens = tokens

        if isinstance(sign_tokens, np.int64):
            sign_tokens = [sign_tokens] * 5

        if self.max_sign_length:
            sign_tokens = sign_tokens[: self.max_sign_length]
        if self.max_token_length:
            sign_tokens = sign_tokens[: self.max_token_length]

        return sign_tokens

    def format_pose(self, keypoints, idx_seed=None):
        """
        Format the pose landmarks based on the data augmentation techniques
        """

        torch.manual_seed(idx_seed + self.seed)

        if self.skip_frames:
            keypoints = keypoints[::2]
        if self.random_drop:
            # Select random frames with the given probability
            random_frames = torch.rand(len(keypoints)) > self.random_drop_rate
            keypoints = keypoints[random_frames]
        if self.random_switch:
            # switch random two frames
            random_frames = torch.randint(0, len(keypoints), (2,))
            keypoints[random_frames[0]], keypoints[random_frames[1]] = (
                keypoints[random_frames[1]],
                keypoints[random_frames[0]],
            )
        if self.duplicate_frames:
            tokens = []
            for token in keypoints:
                tokens.extend([token] * 2)
            keypoints = torch.stack(tokens)

        if isinstance(keypoints, np.int64):
            keypoints = [keypoints] * 5

        if self.max_sign_length:
            keypoints = keypoints[: self.max_sign_length]

        # If object is not a tensor, convert it to tensor
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.tensor(keypoints, dtype=torch.float)

        return keypoints

    def __getitem__(self, index) -> dict:

        # Create random seed from the index
        torch.manual_seed(index)

        if isinstance(index, torch.Tensor):
            index = index.item()
        ####################################### I added error handling for this function !!!!!!!!!!! - fixed
        ########################## Original code :
        if self.shard_id != str(self.idx2instance[index]["shard_id"]):
            self.shard_id = str(self.idx2instance[index]["shard_id"])
            self.h5file = h5py.File(
                os.path.join(
                    self.h5_fpath,
                    ".".join(
                        [self.dataset_id, self.dataset_type, self.shard_id] + ["h5"]
                    ),
                ),
                "r",
            )

        file_id = self.idx2instance[index]["clip_id"]

        if self.use_linear:
            sign_tokens = self.get_pose(file_id)
        else:
            sign_tokens = self.h5file[file_id]["features"][()]
        #################################################################### I added error handling for this function
        #################################################################### This lines were commented *alg
        """
        translation = self.idx2instance[index]["text"]
 
        if self.use_context:
            context = self.idx2instance[index]["context"]
        else:
            context = None

        """
        ##################################################################### *alg
        # NEW: Sample task (SLT or Alignment)
        """
        task_type = sample_task(self.task_weights)
        
        # Force SLT task if we are in dev/test mode or if alignment target is missing/invalid *alg
        if (self.mode != 'train' 
            or task_type == 'sign2text' 
            or 'alignment_target' not in self.idx2instance[index]
            or pd.isna(self.idx2instance[index]['alignment_target'])):
            
            translation = self.idx2instance[index]["text"]
            task_token = '<slt>'
        else: 
            # This is a training-mode alignment task
            translation = self.idx2instance[index]['alignment_target']
            task_token = '<alignment>'

        if self.use_context:
            context = self.idx2instance[index]["context"]
        else:
            context = None
        """
        # Sample task (SLT or Alignment) for ALL modes (train/dev/test)
        #  task_type = sample_task(self.task_weights) *alg
        
        #################################################################### *alg
        # Deterministic flag-based task selection using is_alignment_eligible column
        # Check if is_alignment_eligible column exists and is True
        is_alignment_flag = self.idx2instance[index].get('is_alignment_eligible', False)
        
        # Handle string "True"/"False" values from CSV and pandas boolean
        if isinstance(is_alignment_flag, str):
            is_alignment_flag = is_alignment_flag.strip().lower() == 'true'
        elif pd.isna(is_alignment_flag):
            is_alignment_flag = False
        else:
            is_alignment_flag = bool(is_alignment_flag)
        
        # Determine task type based on is_alignment_eligible flag
        if is_alignment_flag:
            task_type = 'alignment'
            # Alignment task: 
            # Input = clip_text (plain text in prompt)
            # Output = clip_text_with_timestamps (text with timing annotations)
            prompt_text = self.idx2instance[index]['clip_text']
            # Apply ×10 scaling to timestamps for easier learning
            raw_target = self.idx2instance[index]['clip_text_with_timestamps']
            target_text = scale_bracket_timestamps(raw_target, scale_factor=10)
            task_token = '<alignment>'
        else:
            task_type = 'sign2text'
            # SLT task:
            # Input = no text in prompt (only signs)
            # Output = text (standard translation)
            prompt_text = ""
            target_text = self.idx2instance[index]["text"]
            task_token = '<slt>'
        ####################################################################

        # Get context if enabled
        if self.use_context:
            context = self.idx2instance[index].get("context", None)
        else:
            context = None

        #################################################################### *alg
        # Format the sign tokens
        if self.use_linear:
            sign_tokens = self.format_pose(sign_tokens, idx_seed=index)
        else:
            sign_tokens = self.format_sign_tokens(sign_tokens)
        
        if self.idx2instance[index].get("paraphrases") is not None:
            paraphrases = self.idx2instance[index]["paraphrases"]
        else:
            paraphrases = None

        if self.use_bpe or self.use_char:
            input_sequence, target_sequence = self.get_discrete(
                sign_tokens, target_text, context=context, task_token=task_token, prompt_text=prompt_text # Task tokens are used for alg(task_token=task_token) *alg
            )
            return {
                "file_id": file_id,
                "input_sequence": input_sequence,
                "target_sequence": target_sequence,
                "task_type": task_type,
            }

        if self.use_linear:
            (text_inputs, text_attention), (sign_inputs, sign_attention), labels = (
                self.get_continuous(sign_tokens, target_text, context=context, 
                                  task_token=task_token, prompt_text=prompt_text) # Task tokens are used for alg(task_token=task_token) *alg
            )#################################################################################################### *alg
            return {
                "file_id": file_id,
                "text_inputs": text_inputs,
                "sign_inputs": sign_inputs,
                "text_attention_mask": text_attention,
                "sign_attention_mask": sign_attention,
                "sentence": target_text,
                "labels": labels,
                "paraphrases": paraphrases,
                "task_type": task_type, ############################################################### task_type *alg
            }


# class HFDatasetForPretraining(HFDatasetForSLT):

#     def __init__(
#         self,
#         h5_fpath,
#         dataset_id,
#         mode="train",
#         objective="sign2text",
#         max_token_length=128,
#         max_sign_length=250,
#         tokenizer_name="google-t5/t5-small",
#         source_lang="asl",
#         target_lang="en",
#         num_codes=512,
#         annotation_file=None,
#         max_instances=None,
#         skip_frames=True,
#         random_drop=False,
#         random_switch=False,
#         duplicate_frames=False,
#         model_type="t5",
#         inference_mode=False,
#         use_linear=False,
#         use_bpe=False,
#         use_char=True,
#         use_context=False,
#         trim_face=False,
#         seed=42,
#         random_drop_rate=0.1,
#         use_paraphrase=False,
#         dataset_type="main",
#     ):
#         super().__init__(
#             h5_fpath,
#             dataset_id,
#             mode,
#             objective,
#             max_token_length,
#             max_sign_length,
#             tokenizer_name,
#             source_lang,
#             target_lang,
#             num_codes,
#             annotation_file,
#             max_instances,
#             skip_frames,
#             random_drop,
#             random_switch,
#             duplicate_frames,
#             model_type,
#             inference_mode,
#             use_linear,
#             use_bpe,
#             use_char,
#             use_context,
#             trim_face,
#             seed,
#             random_drop_rate,
#             use_paraphrase,
#             dataset_type,
#         )

#     self.max_masked_sequence


class AugmentedDatasetForSLT(Dataset):

    def __init__(self, dataset, augmentations):
        super().__init__()

        self.dataset = dataset
        self.base_dataset = dataset

        print(
            "Size of the base dataset:",
            len(dataset),
            "Size of the COPY base dataset:",
            len(self.base_dataset),
        )
        self.augmentations = augmentations

        self.process_augmentations()

    def process_augmentations(self):

        for augment in self.augmentations:

            aug = augment["name"]

            import inspect

            # Create dummy object from  base dataset parameters
            base_dataset = HFDatasetForSLT(
                **{
                    key: getattr(self.base_dataset, key)
                    for key in inspect.signature(self.base_dataset.__class__).parameters
                }
            )

            if aug == "skip_frames":
                print("Before Skip Frames", len(self.dataset))
                base_dataset.skip_frames = True
                base_dataset.seed = augment["seed"]
                self.dataset = ConcatDataset(
                    [
                        self.dataset,
                        base_dataset,
                    ]
                )
                print("After Skip Frames", len(self.dataset))
                print("-" * 50)
            elif aug == "random_drop":
                print("Before Random Drop", len(self.dataset))
                base_dataset.random_drop = True
                base_dataset.seed = augment["seed"]
                base_dataset.random_drop_rate = augment["prob_drop"]
                self.dataset = ConcatDataset(
                    [
                        self.dataset,
                        base_dataset,
                    ]
                )
                print("After Random Drop", len(self.dataset))
                print("-" * 50)
            elif aug == "random_switch":
                print("Before Random Switch", len(self.dataset))
                base_dataset.random_switch = True
                base_dataset.seed = augment["seed"]
                self.dataset = ConcatDataset(
                    [
                        self.dataset,
                        base_dataset,
                    ]
                )
                print("After Random Switch", len(self.dataset))
                print("-" * 50)
            elif aug == "duplicate_frames":
                print("Before duplication", len(self.dataset))
                base_dataset.duplicate_frames = True
                base_dataset.seed = augment["seed"]
                self.dataset = ConcatDataset(
                    [
                        self.dataset,
                        base_dataset,
                    ]
                )
                print("After duplication", len(self.dataset))
                print("-" * 50)
            else:
                raise ValueError(f"Augmentation {aug} is not supported")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class DatasetForContrastiveTraining(DatasetForPose):

    def __init__(
        self,
        h5_fpath,
        dataset_id,
        mode="train",
        transform="yasl",
        skip_frames=False,
        min_sequence_length=None,
        max_sequence_length=None,
        max_instances=None,
        use_face=False,
        get_origin=False,
    ):
        """
        Dataset for contrastive training. This dataset is used for training the contrastive model

        Args:
            h5_fpath (str): Path to the dataset
            dataset_id (str): Dataset id
            mode (str): Mode of the dataset (train, dev, test)
            transform (str): Transformation to apply to the keypoints
            skip_frames (bool): Skip frames in the sequence
            min_sequence_length (int): Minimum sequence length
            max_sequence_length (int): Maximum sequence length
            max_instances (int): Maximum number of instances to load
            use_face (bool): Use face landmarks
            get_origin (bool): Get the origin of the keypoints

        Returns:
            if get_origin is True:

                file_id (str): File id
                keypoints (torch.Tensor): Keypoints
                sentence (str): Sentence

            else:

                pose_1 (torch.Tensor): Pose 1
                pose_1_aug (torch.Tensor): Augmented pose 1
                pose_2 (torch.Tensor): Pose 2
                sentence_1 (str): Sentence 1
                sentence_2 (str): Sentence 2

        """

        self.mode = mode
        self.use_face = use_face
        self.max_instances = max_instances
        self.get_origin = get_origin
        h5_fpath = os.path.join(h5_fpath, ".".join([dataset_id, "pose", mode] + ["h5"]))

        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

        self.max_sequence_length = max_sequence_length
        self.skip_frames = skip_frames

        super().__init__(h5_fpath, transform, None)
        # Generate pairs for contrastive training if not already generated
        if not self.get_origin:
            self._generate_pairs_(max_samples=max_instances)
        else:
            self.pairs = None

    def augment(
        self,
        x,
        skip_frames=0,
        drop_frames=0,
        mask_pose=0,
        mask_left=0,
        mask_right=0,
        multiply_frames=0,
    ):

        if skip_frames > 0.5:
            x = x[::2, :]

        if drop_frames > 0.5:
            random_indices = np.random.choice(
                x.shape[0], int(x.shape[0] * drop_frames), replace=False
            ).tolist()
            x = x[[i for i in range(x.shape[0]) if i not in random_indices], :]

        if mask_pose > 0.5:
            random_indices = np.random.choice(
                x.shape[0], int(x.shape[0] * mask_pose), replace=False
            ).tolist()

            left_start, left_end = 0, x.shape[1] // 2
            right_start, right_end = x.shape[1] // 2, x.shape[1]

            if mask_right > 0.5:
                x[random_indices, right_start:right_end] = 0
            if mask_left > 0.5:
                x[random_indices, left_start:left_end] = 0
            # If both are not mask whole joint
            if mask_left < 0.5 and mask_right < 0.5:
                x[random_indices, :] = 0

        if multiply_frames > 0.5:
            if x.shape[0] < self.max_sequence_length:
                augmented = []
                for i in range(x.shape[0]):
                    # Copy random times the frame
                    random_times = np.random.randint(1, 5)
                    augmented.extend([x[i]] * random_times)
                x = torch.stack(augmented, dim=0)

        return x

    def _generate_pairs_(self, max_samples=None):
        print("Generating pairs for contrastive training", super().__len__())
        # Generate unique pairs
        self.pairs = torch.randint(0, super().__len__() - 1, (max_samples, 2))
        # Remove if the pairs are the same
        self.pairs = self.pairs[self.pairs[:, 0] != self.pairs[:, 1]]
        # Show unique ids
        print("Number of Unique ids", len(np.unique(self.pairs)))
        self.pairs = self.pairs.tolist()
        print("Number of pairs", len(self.pairs))

    def __len__(self):
        return len(self.pairs) if self.pairs is not None else super().__len__()

    def __getitem__(self, idx):

        # Ensure that the same seed is used for both pairs
        torch.manual_seed(idx)

        # Get the origin of the keypoints
        if self.get_origin:
            file_id, keypoints, sentence = super().__getitem__(idx)
            keypoints = keypoints[:, : -len(FACE_LANDMARKS) * 3]
            return file_id, keypoints, sentence

        pair_1, pair_2 = self.pairs[idx]

        file_id, pose_1, sentence_1 = super().__getitem__(pair_1)
        file_id, pose_2, sentence_2 = super().__getitem__(pair_2)

        # Get crops of the sequence if min_sequence_length and max_sequence_length are provided
        if self.min_sequence_length and self.max_sequence_length:

            # Randomly select a sequence length and start str(index)
            start_idx = torch.randint(
                0, max(pose_1.size(0) - self.min_sequence_length, 1), (1,)
            )
            sequence_length = torch.randint(
                self.min_sequence_length, self.max_sequence_length, (1,)
            )
            # Crop the sequence
            pose_1 = pose_1[start_idx : start_idx + sequence_length]

            start_idx = torch.randint(
                0, max(pose_2.size(0) - self.min_sequence_length, 1), (1,)
            )
            sequence_length = torch.randint(
                self.min_sequence_length, self.max_sequence_length, (1,)
            )

            pose_2 = pose_2[start_idx : start_idx + sequence_length]

        if not self.use_face:
            pose_1 = pose_1[:, : -len(FACE_LANDMARKS) * 3]
            pose_2 = pose_2[:, : -len(FACE_LANDMARKS) * 3]

        pose_1 = pose_1 * 10
        pose_2 = pose_2 * 10

        # Augment the pose
        pose_1_aug = self.augment(
            pose_1,
            **{
                "skip_frames": torch.rand(1).item(),
                "drop_frames": torch.rand(1).item(),
                "mask_pose": torch.rand(1).item(),
                "mask_left": torch.rand(1).item(),
                "mask_right": torch.rand(1).item(),
                "multiply_frames": torch.rand(1).item(),
            },
        )

        return pose_1, pose_1_aug, pose_2, sentence_1, sentence_2
