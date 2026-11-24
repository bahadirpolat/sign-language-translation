# transform how2sign data to h5 format
# NOTE: How2Sign dataset has one signer included for a given video thus `people[0]` is used to extract keypoints, check this for MeineDGS dataset

import os
import json
import h5py
import tarfile
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='Transform How2Sign data to h5 format')
    parser.add_argument(
        '--HOW2SIGN_ROOT',
        type=str,
        default='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/',
        help='Path to How2Sign dataset'
    )
    parser.add_argument(
        '--HOW2SIGN_TARFILE',
        type=str,
        default='{}_2D_keypoints.tar.gz',
        help='How2Sign tar file name'
    )
    parser.add_argument(
        '--HOW2SIGN_CSVFILE',
        type=str,
        default='how2sign_realigned_{}.csv',
        help='How2Sign csv file name'
    )
    parser.add_argument(
        '--HOW2SIGN_JSON_PATH',
        type=str,
        default='{}/openpose_output/json',
        help='How2Sign json path'
    )
    parser.add_argument(
        '--HOW2SIGN_OUTDIR',
        type=str,
        default='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/',
        help='Path to save the h5 file'
    )
    return parser.parse_args()


def transform_pose_data(dataset_type='val'):
    
    assert os.path.exists(os.path.join(HOW2SIGN_ROOT, HOW2SIGN_CSVFILE.format(dataset_type))), "Validation CSV file not found"

    csv_file = pd.read_csv(os.path.join(HOW2SIGN_ROOT, HOW2SIGN_CSVFILE.format(dataset_type)), sep='\t')
    # Format the absolute path of the json files
    csv_file['ABS_PATH'] = csv_file.SENTENCE_NAME.apply(lambda x: os.path.join(HOW2SIGN_ROOT,HOW2SIGN_JSON_PATH.format(dataset_type), x))

    for idx, row in tqdm(csv_file.to_dict(orient='records'), total=csv_file.shape[0]):
        if not os.path.exists(row['ABS_PATH']):
            print('[{}] sentence does not exist:'.format(dataset_type), row['SENTENCE_NAME'])
            continue
        frame_files = os.listdir(row['ABS_PATH'])
        frame_files = sorted(frame_files, key=lambda x: int(x.split('_')[-2].split('.')[0])) 

        sentence_keypoints = {
            'face_keypoints': [],
            'pose_keypoints': [],
            'right_hand_keypoints': [],
            'left_hand_keypoints': []
        }
        for frame in frame_files:
            with open(os.path.join(row['ABS_PATH'], frame)) as f:
                frame_data = json.load(f)
                
                pose_keypoints = np.array(
                    frame_data['people'][0]['pose_keypoints_2d']
                ).reshape(-1, 3).astype(float)

                right_hand_keypoints = np.array(
                    frame_data['people'][0]['hand_right_keypoints_2d']
                ).reshape(-1, 3).astype(float)

                left_hand_keypoints = np.array(
                    frame_data['people'][0]['hand_left_keypoints_2d']
                ).reshape(-1, 3).astype(float)

                face_keypoints = np.array(
                    frame_data['people'][0]['face_keypoints_2d']
                ).reshape(-1, 3).astype(float)

                sentence_keypoints['face_keypoints'].append(face_keypoints)
                sentence_keypoints['pose_keypoints'].append(pose_keypoints)
                sentence_keypoints['right_hand_keypoints'].append(right_hand_keypoints)
                sentence_keypoints['left_hand_keypoints'].append(left_hand_keypoints)
        
        sentence_keypoints['face_keypoints'] = np.array(sentence_keypoints['face_keypoints'])
        sentence_keypoints['pose_keypoints'] = np.array(sentence_keypoints['pose_keypoints'])
        sentence_keypoints['right_hand_keypoints'] = np.array(sentence_keypoints['right_hand_keypoints'])
        sentence_keypoints['left_hand_keypoints'] = np.array(sentence_keypoints['left_hand_keypoints'])

        with h5py.File(os.path.join(HOW2SIGN_OUTDIR,'how2sign_{}.h5'.format(dataset_type)), 'a') as hf:
            if hf.get(row['SENTENCE_NAME']): # if sentence already exists, skip
                continue
            hf.create_group(row['SENTENCE_NAME'])
            hf[row['SENTENCE_NAME']].create_dataset('face_keypoints', data=sentence_keypoints['face_keypoints'])
            hf[row['SENTENCE_NAME']].create_dataset('pose_keypoints', data=sentence_keypoints['pose_keypoints'])
            hf[row['SENTENCE_NAME']].create_dataset('right_hand_keypoints', data=sentence_keypoints['right_hand_keypoints'])
            hf[row['SENTENCE_NAME']].create_dataset('left_hand_keypoints', data=sentence_keypoints['left_hand_keypoints'])
            hf[row['SENTENCE_NAME']].create_dataset('sentence', data=row['SENTENCE'])


if __name__ == '__main__':

    args = parse_args()

    # Set global variables
    global HOW2SIGN_ROOT, HOW2SIGN_TARFILE, HOW2SIGN_CSVFILE, HOW2SIGN_JSON_PATH, HOW2SIGN_OUTDIR
    
    HOW2SIGN_ROOT = args.HOW2SIGN_ROOT
    HOW2SIGN_TARFILE = args.HOW2SIGN_TARFILE
    HOW2SIGN_CSVFILE = args.HOW2SIGN_CSVFILE
    HOW2SIGN_JSON_PATH = args.HOW2SIGN_JSON_PATH
    HOW2SIGN_OUTDIR = args.HOW2SIGN_OUTDIR

    tar = tarfile.open(os.path.join(HOW2SIGN_ROOT, HOW2SIGN_TARFILE.format('train')), "r:gz")
    tar.extractall(path=os.path.join(HOW2SIGN_ROOT, 'train'))

    tar = tarfile.open(os.path.join(HOW2SIGN_ROOT, HOW2SIGN_TARFILE.format('val')), "r:gz")
    tar.extractall(path=os.path.join(HOW2SIGN_ROOT, 'val'))

    tar = tarfile.open(os.path.join(HOW2SIGN_ROOT, HOW2SIGN_TARFILE.format('test')), "r:gz")
    tar.extractall(path=os.path.join(HOW2SIGN_ROOT, 'test'))

    transform_pose_data('train')
    transform_pose_data('val')
    transform_pose_data('test')