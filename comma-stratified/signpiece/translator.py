
import torch
import numpy as np
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

# SignPiece imports
from signpiece import SignT5
from signpiece import SignT5Config
from signpiece.utils.transforms import normalize_yasl

import cv2
import mediapipe as mp
mp_holistic = mp.solutions.holistic

from typing import List, Dict, Any

# Landmarks for the upper body
POSE_LANDMARKS = [
    0, 1, 2, 3, 4, 5, 
    6, 7, 8, 9, 10, 
    11, 12, 13, 14, 
    15, 16, 17, 18, 19, 
    20, 21, 22, 23, 24,
]

# Face landmarks (Should have all 468 landmarks)
FACE_LANDMARKS = [
    0, 4, 13, 14, 17, 33, 37, 39, 46,
    52, 55, 61, 64, 81, 82, 93,
    133, 151, 152, 159, 172, 178, 181, 
    263, 269, 276, 282, 285, 291, 294,
    311, 323, 362, 386, 397,
    # 468, 473 
]

class SignpieceTranslator():

    def __init__(self, model_path: str):
        self.model = SignT5.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config.base_model_name)

        # From generation arguments
        self.max_length = 128
        self.num_beams = 4
        self.length_penalty = 0.6
        self.early_stopping = True
        self.no_repeat_ngram_size = 2
        self.decoder_start_token_id = self.tokenizer.pad_token_id
        self.num_return_sequences = 1

        # From sebeder inference
        self.min_detection_confidence = 0.1
        self.min_tracking_confidence = 0.1
        self.model_complexity = 2
        self.max_float_decimal = 3


    @torch.no_grad()
    def translate(self,
                  sign_inputs: List[torch.Tensor],
                  sign_input_lengths: List[int],) -> Dict[str, Any]:
        
        # Create attention mask for the input lengths
        sign_attention_mask = pad_sequence([torch.ones(length) for length in sign_input_lengths], batch_first=True, padding_value=0)

        outputs = self.model.generate(sign_inputs=sign_inputs,
                                      sign_attention_mask=sign_attention_mask,
                                      text_inputs=None,
                                      text_attention_mask=None,
                                      max_length=self.max_length,
                                      num_beams=self.num_beams,
                                      length_penalty=self.length_penalty,
                                      early_stopping=self.early_stopping,
                                      no_repeat_ngram_size=self.no_repeat_ngram_size,
                                      decoder_start_token_id=self.decoder_start_token_id,
                                      num_return_sequences=self.num_return_sequences)

        # Decode the output
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            "translated_text": decoded_outputs
        }
    
    def translate_video(self, video_path: str) -> Dict[str, Any]:
        """
        Translate the sign language from the video
        
        Args:
            video_path (str): Path to the video
            
        Returns:
            Dict[str, Any]: Translated text
        """
        # Get the pose landmarks
        pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks = self._get_pose_from_video(video_path)
        # Format the pose landmarks for the model input
        model_input = self._format_pose_for_model_input(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks)
        # Translate the input
        translation = self.translate(model_input.unsqueeze(0), [model_input.size(0)])
        return translation
    
    def translate_live(self) -> Dict[str, Any]:
        """
        Translate the sign language from the webcam video
        
        Returns:
            Dict[str, Any]: Translated text
        """
        # This version gets landmarks from the webcam until the stop condition is met
        pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks = self._get_pose_from_video(0, use_stop_condition=True)
        # Format the pose landmarks for the model input
        model_input = self._format_pose_for_model_input(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks)
        # Translate the input
        translation = self.translate(model_input.unsqueeze(0), [model_input.size(0)])

        return translation
        
    def _get_pose_from_video(self, video_path: str, use_stop_condition: bool = False) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks = [], [], [], []
        while cap.isOpened():
            with mp_holistic.Holistic(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=self.model_complexity
            ) as holistic:

                ret, frame = cap.read()

                if not ret:
                    break

                # Convert the BGR image to RGB.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the image and extract pose landmarks.
                results = holistic.process(image)

                # Extract the landmarks
                pose_landmark, left_hand_landmark, right_hand_landmark, face_landmark = None, None, None, None
                if results.pose_landmarks:
                    pose_landmark = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]).flatten()
                if results.left_hand_landmarks:
                    left_hand_landmark = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.left_hand_landmarks.landmark]).flatten()
                if results.right_hand_landmarks:
                    right_hand_landmark = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark]).flatten()
                if results.face_landmarks:
                    face_landmark = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.face_landmarks.landmark]).flatten()
                
                # Check stop condition if given 
                # TODO: Implement stop condition
                if use_stop_condition:
                    # The stop condition is when the arms are lowered in last 10 frames
                    if len(pose_landmarks) > 10:
                        if np.all(pose_landmarks[-10:, 11, 1] > pose_landmarks[-10:, 23, 1]):
                            break

                # Append the landmarks if they exist, otherwise append zeros
                pose_landmarks.append(np.array(pose_landmark) if pose_landmark else np.zeros((33, 3)))
                left_hand_landmarks.append(np.array(left_hand_landmark) if left_hand_landmark else np.zeros((21, 3)))
                right_hand_landmarks.append(np.array(right_hand_landmark) if right_hand_landmark else np.zeros((21, 3)))
                face_landmarks.append(np.array(face_landmark) if face_landmark else np.zeros((468, 3)))

        cap.release()

        # Convert the landmarks to numpy arrays
        pose_landmarks = np.array(pose_landmarks)
        left_hand_landmarks = np.array(left_hand_landmarks)
        right_hand_landmarks = np.array(right_hand_landmarks)
        face_landmarks = np.array(face_landmarks)

        # Get max decimal points of the float values
        pose_landmarks = np.round(pose_landmarks, self.max_float_decimal)
        left_hand_landmarks = np.round(left_hand_landmarks, self.max_float_decimal)
        right_hand_landmarks = np.round(right_hand_landmarks, self.max_float_decimal)
        face_landmarks = np.round(face_landmarks, self.max_float_decimal)

        return pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks
    
    def _format_pose_for_model_input(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks):
        
        # Normalize the landmarks
        pose_landmarks = normalize_yasl(pose_landmarks)
        left_hand_landmarks = normalize_yasl(left_hand_landmarks)
        right_hand_landmarks = normalize_yasl(right_hand_landmarks)
        face_landmarks = normalize_yasl(face_landmarks)

        # Trim selected landmarks
        pose_landmarks = pose_landmarks[:, POSE_LANDMARKS, :]
        face_landmarks = face_landmarks[:, FACE_LANDMARKS, :]

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
    



