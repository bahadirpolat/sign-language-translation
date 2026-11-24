import os
import h5py
import json
import webvtt
import requests
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()

BOBSL_PREFIX = "https://thor.robots.ox.ac.uk/bobsl/v1.3/pose/"
BOBSL_SUFFIX = ".tar.gz"

BOBSL_USERNAME = os.getenv("BOBSL_USERNAME")
BOBSL_PASSWORD = os.getenv("BOBSL_PASSWORD")

def download_file(url, local_filename):

    with requests.get(url, stream=True, auth=(BOBSL_USERNAME, BOBSL_PASSWORD)) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")

def download_bobsl_file(file_id, save_dir="."):
    url = BOBSL_PREFIX + file_id + BOBSL_SUFFIX
    local_filename = file_id + BOBSL_SUFFIX
    download_file(url, os.path.join(save_dir, local_filename))
    
def extract_bobsl_file(file_id, save_dir="."):
    local_filename = file_id + BOBSL_SUFFIX
    # create a directory with the same name as the file
    os.system(f"mkdir -p {os.path.join(save_dir, file_id)}")
    os.system(f"tar -xzf {os.path.join(save_dir, local_filename)} -C {os.path.join(save_dir, file_id)}")


def get_subtitles(episode_id, annotation_type='automatic_annotations', subtitle_type='audio_aligned_heuristic_correction', fps=25, subtitle_dir="data/"):
    subtitles = []
    
    SUBTITLE_DIR = f'{subtitle_dir}/bobsl/v1.4/{annotation_type}/signing_aligned_subtitles/'
    
    # check if the file exists
    if not os.path.exists(os.path.join(SUBTITLE_DIR, subtitle_type ,episode_id + ".vtt")):
        print(f"Subtitle file for episode {episode_id} does not exist")
        print((os.path.join(SUBTITLE_DIR, subtitle_type ,episode_id + ".vtt")))
        return subtitles
    for caption in webvtt.read(os.path.join(SUBTITLE_DIR, subtitle_type, episode_id + ".vtt")):
        # convert the start and end time to milliseconds
        import re 

        hours, minutes, seconds, ms = map(int, re.split(r':|\.', caption.start))
        start_in_ms = hours * 3600000 + minutes * 60000 + seconds * 1000 + ms

        hours, minutes, seconds, ms = map(int, re.split(r':|\.', caption.end))
        end_in_ms = hours * 3600000 + minutes * 60000 + seconds * 1000 + ms

        start_in_frame = int(start_in_ms * fps / 1000)
        end_in_frame = int(end_in_ms * fps / 1000)
        
        subtitles.append({
            "episode_id": episode_id,
            "start": caption.start,
            "end": caption.end,
            "start_in_ms": start_in_ms,
            "end_in_ms": end_in_ms,
            # Calculate start and end in frames based on the frame rate of the video
            # FPS: 30
            "start_in_frame": start_in_frame,
            "end_in_frame": end_in_frame,
            "text": caption.text
        })

    return subtitles

def extract_dataset(dataset, save_dir="data/", dump_dir='data', fps=25, subset="train"):

    for idx, episode_id in tqdm(list(enumerate(dataset.episode_id.unique()))):

        # first extract keypoint dataset
        download_bobsl_file(episode_id, save_dir=dump_dir)
        extract_bobsl_file(episode_id, save_dir=dump_dir)

        keypoint_length = len(os.listdir(os.path.join(dump_dir, episode_id)))
        max_length = dataset[dataset.episode_id == episode_id].end_in_frame.max()
        if keypoint_length < max_length:
            print(keypoint_length, max_length)
            print(f"Episode {episode_id} has less keypoint files than expected")
            continue

        for (_, annotation) in tqdm(dataset[dataset.episode_id == episode_id].iterrows(), desc=f"Processing episode {episode_id}"):

            pose, left_hand, right_hand, face = [], [], [], []
            for frame in range(annotation.start_in_frame, annotation.end_in_frame):

                with open(os.path.join(dump_dir, episode_id, f'{episode_id}_{str(frame).zfill(12)}_keypoints.json')) as f:
                    data = json.load(f)

                if len(data['people']) == 0:
                    continue

                # get the keypoints for the frame
                pose_keypoints = data['people'][0]['pose_keypoints_2d']
                left_hand_keypoints = data['people'][0]['hand_left_keypoints_2d']
                right_hand_keypoints = data['people'][0]['hand_right_keypoints_2d']
                face_keypoints = data['people'][0]['face_keypoints_2d']

                pose.append(np.array(pose_keypoints).reshape(-1, 3))
                left_hand.append(np.array(left_hand_keypoints).reshape(-1, 3))
                right_hand.append(np.array(right_hand_keypoints).reshape(-1, 3))
                face.append(np.array(face_keypoints).reshape(-1, 3))

            if len(pose) == 0:
                print(f"No keypoints found for episode {episode_id}-{annotation.start_in_frame}-{annotation.end_in_frame}")
                continue

            pose = np.array(pose)
            left_hand = np.array(left_hand)
            right_hand = np.array(right_hand)
            face = np.array(face)

            with h5py.File(os.path.join(save_dir, f'bobsl.pose.{subset}.h5'), 'a') as f:
                clip_id = f'{episode_id}_{annotation.start_in_frame}_{annotation.end_in_frame}'
                f.create_group(clip_id)
                f.create_group(f'{clip_id}/joints')
                f.create_dataset(f'{clip_id}/joints/pose', data=pose)
                f.create_dataset(f'{clip_id}/joints/left_hand', data=left_hand)
                f.create_dataset(f'{clip_id}/joints/right_hand', data=right_hand)
                f.create_dataset(f'{clip_id}/joints/face', data=face)
                f.create_dataset(f'{clip_id}/sentence', data=annotation.text)

        os.remove(os.path.join(dump_dir, episode_id + BOBSL_SUFFIX))
        os.system(f"rm -rf {os.path.join(dump_dir, episode_id)}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Extract BOBSL dataset")
    parser.add_argument("--dataset_dir", type=str, default="/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/BOBSL/subset2episode.json")
    parser.add_argument("--save_dir", type=str, default="data/")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--subset", type=str, default="train", choices=["train", "dev", "test"])
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    with open(os.path.join(args.dataset_dir, 'subset2episode.json')) as f: data = json.load(f)

    train_episodes = data['train']
    test_episodes = data['test']
    dev_episodes = data['val']

    train_annotations = []
    for episode_id in tqdm(train_episodes, desc="Train Episodes"):
        train_annotations += get_subtitles(episode_id, annotation_type='automatic_annotations', subtitle_dir=args.dataset_dir)

    dev_annotations = []
    for episode_id in tqdm(dev_episodes, desc="Dev Episodes"):
        dev_annotations += get_subtitles(episode_id, annotation_type='automatic_annotations', subtitle_dir=args.dataset_dir)

    test_annotations = []
    for episode_id in tqdm(test_episodes, desc="Test Episodes"):
        test_annotations += get_subtitles(episode_id, annotation_type='manual_annotations', subtitle_type='', subtitle_dir=args.dataset_dir)

    train_df = pd.DataFrame(train_annotations)
    dev_df = pd.DataFrame(dev_annotations)
    test_df = pd.DataFrame(test_annotations)



    # do it 16 threads
    import multiprocessing

    num_threads = 16
    # create 16 train splits
    train_df_splits = np.array_split(train_df, num_threads)
    pool = multiprocessing.Pool(num_threads)
    pool.starmap(extract_dataset, [
        (train_df_split, args.save_dir, args.dataset_dir, args.fps, f"train.{idx}") for idx, train_df_split in enumerate(train_df_splits)
    ])
    pool.close()
    pool.join()

    dev_df_splits = np.array_split(dev_df, num_threads)
    pool = multiprocessing.Pool(num_threads)
    pool.starmap(extract_dataset, [
        (dev_df_split, args.save_dir, args.dataset_dir, args.fps, f"dev.{idx}") for idx, dev_df_split in enumerate(dev_df_splits)
    ])
    pool.close()
    pool.join()

    test_df_splits = np.array_split(test_df, num_threads)
    pool = multiprocessing.Pool(num_threads)
    pool.starmap(extract_dataset, [
        (test_df_split, args.save_dir, args.dataset_dir, args.fps, f"test.{idx}") for idx, test_df_split in enumerate(test_df_splits)
    ])
    pool.close()
    pool.join()

    
    # extract_dataset(train_df, subset="train", save_dir=args.save_dir)
    # extract_dataset(dev_df, subset="dev", save_dir=args.save_dir)
    # extract_dataset(test_df, subset="test", save_dir=args.save_dir)