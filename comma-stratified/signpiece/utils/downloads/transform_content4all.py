# Description: Load the openpose data and annotations from the Content4All dataset and save them to an h5 file
# TODO: Add download_from_url function to download the openpose data from the URL
# OPENPOSE ANNOTATIONS
# wget --user $1 --password $2 "https://www.cvssp.org/data/c4a-news-corpus/annotated-data/swisstxt-news/openpose2D/swisstxt-news-openpose2D.tar.gz"
# wget --user $1 --password $2 "https://www.cvssp.org/data/c4a-news-corpus/annotated-data/swisstxt-weather/openpose2D/swisstxt-weather-openpose2D.tar.gz"
# wget --user $1 --password $2 "https://www.cvssp.org/data/c4a-news-corpus/annotated-data/vrt-news/openpose2D/vrt-news-openpose2D.tar.gz"
# SUBTITLES ANNOTATIONS
# wget --user $1 --password $2 "https://www.cvssp.org/data/c4a-news-corpus/annotated-data/swisstxt-news/swisstxt-news-annotations.json"
# wget --user $1 --password $2 "https://www.cvssp.org/data/c4a-news-corpus/annotated-data/swisstxt-weather/swisstxt-weather-annotations.json"
# wget --user $1 --password $2 "https://www.cvssp.org/data/c4a-news-corpus/annotated-data/vrt-news/vrt-news-annotations.json"

# NOTE: Check out BODY_135 from https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/41d8c087459fae844e477dda50a6f732e70f2cb8/src/openpose/pose/poseParameters.cpp#L149

import os
import h5py
import json
import tarfile
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_openpose_data(
    path, 
    annotations,
    output_dir,
    dataset_id
)-> None:
    """
    Load the openpose data and annotations and save them to an h5 file
    
    Args:
        path (str): Path to the openpose data
        annotations (pd.DataFrame): Annotations for the videos
        output_dir (str): Path to save the h5 file
        dataset_id (str): ID for the dataset
    """    
    tmp_path = os.path.join(args.dataset_dir, 'tmp')
    os.makedirs(tmp_path, exist_ok=True)

    for file in tqdm(os.listdir(path), desc='Processing episodes'):

        video_id = file.split('.')[0]

        video_annotations = annotations[annotations.video_id == video_id]
        if video_annotations.empty:
            print(f"No annotations found for {video_id}")
            continue

        os.makedirs(output_dir, exist_ok=True)
        f =  h5py.File(os.path.join(output_dir, f"{dataset_id}.h5"), 'a')
        
        def read_file():
            tmp_pose = os.path.join(tmp_path, video_id)
            os.makedirs(tmp_pose, exist_ok=True)

            print(f"Extracting {file} to {tmp_pose}")
            print('Extracting', os.path.join(path, file))
            
            if not os.listdir(tmp_pose):
                tarfile.open(os.path.join(path, file), 'r:xz').extractall(tmp_pose)
            else: print('Skipping extraction')

            sorted_files = sorted(os.listdir(tmp_pose))

            print(video_annotations.sort_values(by='stop_frame',ascending=False).head(10).to_markdown())

            return f, tmp_pose, sorted_files
        
        tmp_pose, sorted_files = None, None

        for annotation in tqdm(video_annotations.to_dict(orient='records'), desc='Processing annotations'):

            clip_id = f'{video_id}_{annotation["start_frame"]}_{annotation["stop_frame"]}'
            if clip_id in f.keys():
                # print(f"Skipping {clip_id} as it already exists")
                continue

            if sorted_files is None:
                f, tmp_pose, sorted_files = read_file()
            
            start_frame, stop_frame = annotation['start_frame'], annotation['stop_frame']

            pose_keypoints, left_hand_keypoints, right_hand_keypoints, face_keypoints = [], [], [], []
            stop_frame = min(stop_frame, len(sorted_files))
            for frame in range(start_frame, stop_frame):
                frame_file = os.path.join(tmp_pose, f"{video_id}_{frame:012}_keypoints.json")
                with open(os.path.join(tmp_pose, frame_file), 'rt') as rt:
                    
                    data = json.loads(rt.read())
                    # FIXME: Some frames have no people detected in the video for that add a check if the data['people'] is empty add np.zeros(75, 3) to the keypoints
                    if not data['people']:
                        # pose_keypoints.append(np.zeros((25, 3)))
                        # left_hand_keypoints.append(np.zeros((21, 3)))
                        # right_hand_keypoints.append(np.zeros((21, 3)))
                        # face_keypoints.append(np.zeros((70, 3)))
                        continue

                    all_keypoint = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3).astype(float)

                    pose_keypoint = all_keypoint[:25]
                    # NOTE: Add the wrist keypoints to the hand keypoints to make 21 keypoints
                    left_hand_keypoint = np.concatenate([all_keypoint[None,9,:], all_keypoint[25:45,:]])
                    right_hand_keypoint = np.concatenate([all_keypoint[None,10,:], all_keypoint[45:65,:]])
                    face_keypoint = all_keypoint[65:]

                    pose_keypoints.append(pose_keypoint)
                    left_hand_keypoints.append(left_hand_keypoint)
                    right_hand_keypoints.append(right_hand_keypoint)
                    face_keypoints.append(face_keypoint)

            # Write to h5 file
            # TODO: Migrate writing to h5 file to a separate function from the one in crawl_publicdgs.py
            f.create_group(clip_id)
            f.create_group(clip_id + '/joints')
            f[f'{clip_id}/joints'].create_dataset('pose_keypoints', data=np.array(pose_keypoints))
            f[f'{clip_id}/joints'].create_dataset('left_hand_keypoints', data=np.array(left_hand_keypoints))
            f[f'{clip_id}/joints'].create_dataset('right_hand_keypoints', data=np.array(right_hand_keypoints))
            f[f'{clip_id}/joints'].create_dataset('face_keypoints', data=np.array(face_keypoints))
            f[f'{clip_id}'].create_dataset('sentence', data=annotation['annotation'])
            # add metadata
            f[clip_id].create_group('metadata')
            for key, value in annotation.items():
                f[clip_id]['metadata'].create_dataset(key, data=value)

        # remove the extracted files
        os.system(f'rm -rf {tmp_pose}')


def eval_split(dataset_dir, dataset_id):

    import h5py
    from tqdm import tqdm

    swisstxt_news = h5py.File(os.path.joint(dataset_dir, f'{dataset_id}.h5'), 'r')

    from sklearn.model_selection import train_test_split
    # split the data into training (80%) and validation (10%) and test (10%) sets
    train_dataset, test_dataset = train_test_split(
        list(swisstxt_news.keys()),
        test_size=0.2,
    )

    val_dataset, test_dataset = train_test_split(
        test_dataset,
        test_size=0.5,
    )

    # write keys and the content into 'train', 'val', and 'test' h5 files with prefix 'swisstxt-news'
    def write_partition(dataset, data_path, mode='train'):
        import os   
        fpath = f'{data_path}/swisstxt-news.{mode}.h5'
        os.remove(fpath) if os.path.exists(fpath) else print('Creating data file')
        h5 = h5py.File(fpath, 'a')
        for _id in tqdm(dataset):
            for item in swisstxt_news[_id].items():
                # check if item is group or dataset
                if item[1].__class__ == h5py._hl.group.Group:
                    for _it in item[1].items():
                        h5.create_dataset(_id + '/' + item[0] + '/' + _it[0], data=_it[1][()])
                else:
                    h5.create_dataset(_id + '/' + item[0], data=item[1][()])

    print(f'** [{dataset_id}] Train:', len(train_dataset))
    write_partition(train_dataset, 'train')
    print(f'** [{dataset_id}] Val:', len(val_dataset))
    write_partition(val_dataset, 'val')
    print(f'** [{dataset_id}] Test:', len(test_dataset))
    write_partition(test_dataset, 'test')


def parse_args():

    import argparse
    parser = argparse.ArgumentParser(description="Load the openpose data and annotations from the Content4All dataset and save them to an h5 file")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/Content4All",
        help="Path to the Content4All dataset"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/Content4All/processed',
        help='Path to save the h5 file'
    )
    return parser.parse_args()

if __name__ == '__main__':

    POSE_PATH = '{}/ANNOTATED-DATASETS/{}/OpenPose2D'

    args = parse_args()

    SWISS_NEWS = POSE_PATH.format(args.dataset_dir,'SWISSTXT-NEWS')
    SWISS_WEATHER = POSE_PATH.format(args.dataset_dir,'SWISSTXT-WEATHER')
    VRT_NEWS = POSE_PATH.format(args.dataset_dir,'VRT-NEWS')    

    if not os.path.exists(SWISS_NEWS):
        tarfile.open(os.path.join(args.dataset_dir, 'swisstxt-news-openpose2D.tar.gz'), 'r:gz').extractall(args.dataset_dir)
    if not os.path.exists(SWISS_WEATHER):
        tarfile.open(os.path.join(args.dataset_dir, 'swisstxt-weather-openpose2D.tar.gz'), 'r:gz').extractall(args.dataset_dir)
    if not os.path.exists(VRT_NEWS):
        tarfile.open(os.path.join(args.dataset_dir, 'vrt-news-openpose2D.tar.gz'), 'r:gz').extractall(args.dataset_dir)

    SWISS_NEWS_ANNOTATIONS = f'{args.dataset_dir}/swisstxt-news-annotations.json'
    SWISS_WEATHER_ANNOTATIONS = f'{args.dataset_dir}/swisstxt-weather-annotations.json'
    VRT_NEWS_ANNOTATIONS = f'{args.dataset_dir}/vrt-news-annotations.json'

    swisstxt_news_annotations = json.load(open(SWISS_NEWS_ANNOTATIONS, 'r'))
    swisstxt_news_annotations = pd.DataFrame.from_records(list(swisstxt_news_annotations.values()))

    swisstxt_weather_annotations = json.load(open(SWISS_WEATHER_ANNOTATIONS, 'r'))
    swisstxt_weather_annotations = pd.DataFrame.from_records(list(swisstxt_weather_annotations.values()))

    vrt_news_annotations = json.load(open(VRT_NEWS_ANNOTATIONS, 'r'))
    vrt_news_annotations = pd.DataFrame.from_records(list(vrt_news_annotations.values()))

    load_openpose_data(
        path=SWISS_NEWS,
        annotations=swisstxt_news_annotations,
        output_dir=args.output_dir,
        dataset_id='swisstxt-news'
    )

    load_openpose_data(
        path=SWISS_WEATHER,
        annotations=swisstxt_weather_annotations,
        output_dir=args.output_dir,
        dataset_id='swisstxt-weather'
    )

    eval_split(args.output_dir, 'swisstxt-news')

    load_openpose_data(
        path=VRT_NEWS,
        annotations=vrt_news_annotations,
        output_dir=args.output_dir,
        dataset_id='vrt-news'
    )

    eval_split(args.output_dir, 'vrt-news')