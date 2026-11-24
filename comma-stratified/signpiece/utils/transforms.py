import numpy as np

def center_norm(keypoints):
    # NOTE: Some points are (0,0) which will may lead to errors in the normalization
    # Also check points that are confidences
    mean = np.mean(keypoints, axis=0)
    # plt.scatter(mean[0], mean[1], c='b')
    # get mean point as the center of the hand and normalize the points around it
    keypoints = keypoints - mean
    # add min and max values to each axis
    max_x, max_y, max_z = np.max(np.abs(keypoints), axis=0)
    # normalize the points
    keypoints = keypoints / max([max_x, max_y])
    return keypoints


def normalize_yasl(landmarks):
    """
    Normalize the keypoints of the YASL dataset.

    Args:
        keypoints (List(np.ndarray)): The keypoints to normalize.
        keypoint_type (str): The type of keypoints to normalize.

    Returns:
        np.ndarray: The normalized keypoints.
    """
    # represent landmarks that are not present in a frame with a large negative value
 
    # if one row is all zeros, the frame is empty -> set all values to -1
    # landmarks[landmarks[:,:,:2].sum(axis=2) == 0] = -1
    zero_rows = np.where(landmarks[:,:,:2].sum(axis=2) == 0)

    min_x_y = np.min(landmarks[:, :, :2], axis=1)
    max_x_y = np.max(landmarks[:, :, :2], axis=1)

    # Normalize the keypoints
    landmarks[:,:,:2] = (landmarks[:,:,:2] - min_x_y[:, None, :]) / (
        max_x_y - min_x_y + 1e-8
    )[:, None, :]

    # Set the landmarks that were empty to -1
    landmarks[zero_rows] = -1

    return landmarks


def get_active_frames(pose_landmarks):
    """
    Get the active frames of the YASL dataset. Either right or left hand is higher than middle point.
    The active frames are the ones where the hand is above the middle point.
    The middle point is the average of the y-coordinates of the shoulder and waist.
    The active frames are the ones where the hand is above the middle point.

    Args:
        pose_landmarks (np.ndarray): The keypoints to normalize.
        
    Returns:
        np.ndarray: The active frames.
    """

    # Get the y-coordinates of the shoulder and waist
    shoulder_y = (pose_landmarks[:, 11, 1] + pose_landmarks[:, 12, 1]) / 2
    waist_y = (pose_landmarks[:, 23, 1] + pose_landmarks[:, 24, 1]) / 2

    # Get the middle point
    middle_point = (shoulder_y + waist_y) / 2

    # Get the active frames if the hand is above the middle point
    left_hand_y = pose_landmarks[:, 15, 1]
    right_hand_y = pose_landmarks[:, 16, 1]
    
    left_hand_active = np.where(left_hand_y > middle_point)[0]
    right_hand_active = np.where(right_hand_y > middle_point)[0]
    
    if len(left_hand_active) == 0:
        left_hand_active = np.array([0, len(pose_landmarks)])
    if len(right_hand_active) == 0:
        right_hand_active = np.array([0, len(pose_landmarks)])
    
    min_active = np.minimum(left_hand_active[0], right_hand_active[-1])
    max_active = np.maximum(left_hand_active[0], right_hand_active[-1])

    return min_active, max_active
    
    