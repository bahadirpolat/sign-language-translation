import os
import h5py
import gzip
import json
import requests
import numpy as np
from pympi.Elan import Eaf
from bs4 import BeautifulSoup
from tqdm import tqdm

DGS_URL = 'https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_en.html'
DATAPATH = '/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/DGS'

def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='Crawl DGS dataset')
    parser.add_argument(
        '--mode', 
        type=str, 
        default='train', 
        help='train or test'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='output_file.gz',
        help='Output file path'
    )
    parser.add_argument(
        '--url',
        type=str,
        default='https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_en.html',
        help='URL to crawl'
    )
    parser.add_argument(
        '--datapath',
        type=str,
        default='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/DGS',
        help='Path to save the data'
    )

    return parser.parse_args()


def download_gz(url, output_file):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Path where you want to save the .gz file
    output_file = "output_file.gz"

    # Write the response content to a file
    with open(output_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # Filter out keep-alive new chunks
                file.write(chunk)

    print(f"File downloaded and saved as {output_file}")

def crawl_dgs(mode='train'):

    args = parse_args()

    # Get the DGS website
    response = requests.get(args.URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get each transcript
    ITEMS = [ td.find_all('td', class_='ilink') for td in soup.find_all('tr')[1:]]
    METADATA = [ get_metadata(td.find_all('td')[:4]) for td in soup.find_all('tr')[1:]]

    def get_metadata(item):
        if not item:
            return
        transcript_id = item[0].text
        age_group = item[1].text
        format_ = [ a.text for a in item[2].find_all('a')]
        tags = [ a.text for a in item[3].find_all('a')]

        return transcript_id, age_group, format_, tags

    assert len(ITEMS) == len(METADATA), 'Number of items and metadata do not match'    
    
    DATA = []
    for ITEM, MD in zip(ITEMS, METADATA):
        doc_id = ITEM[0].find('a').get('download').split('.')[0]
        sources = {}
        for item in ITEM:
            attr = item.find('a').find('img').get('alt')
            url = 'https://www.sign-lang.uni-hamburg.de/meinedgs/'+'/'.join(item.find('a').get('href').split('/')[1:])
            sources[attr] = url

        sources['transcript'] = f'https://www.sign-lang.uni-hamburg.de/meinedgs/{MD[0]}.html'

        DATA.append({
            'doc_id': doc_id,
            'sources': sources,
            'metadata': MD
        })

    # Add crawling openpose data and save to h5 file according to the subtitles from elan
    # Step 1: Download data['sources']['OpenPose']
    # Step 2: Download data['sources']['ELAN']
    # Step 3: Save ELAN data tmp file and read using pympi.Elan

    for data in tqdm(DATA):

        file_id = data['doc_id']
        elan_html = requests.get(data['sources']['ELAN'])

        with open(f'{DATAPATH}/{file_id}.eaf', 'w') as f:
            f.write(elan_html.text) 

        elan_file = Eaf(f'{DATAPATH}/{file_id}.eaf')

        tier_a = elan_file.get_annotation_data_for_tier('Deutsche_Übersetzung_A')
        tier_b = elan_file.get_annotation_data_for_tier('Deutsche_Übersetzung_B')

        # tier_a has entries (start_ms, end_ms, annotation)
        # tier_b has entries (start_ms, end_ms, annotation)
        # translate them into frame numbers for 50fps video
        def ms_to_frame(ms, fps=50): return int((ms / 1000) * fps)

        tier_a_frames = [
            (ms_to_frame(start), ms_to_frame(end), annotation) for start, end, annotation in tier_a
        ]
        tier_b_frames = [
            (ms_to_frame(start), ms_to_frame(end), annotation) for start, end, annotation in tier_b
        ]

        # write openpose
        download_gz(data['sources']['OpenPose'], f'{DATAPATH}/{file_id}.json.gz')

        with gzip.open(f'{DATAPATH}/{file_id}.json.gz', mode="rt") as f:
            data = f.read()
            data = json.loads(data)

        # get poses from openpose data
        video_a = list(data[0]['frames'].values())
        video_b = list(data[1]['frames'].values())

        write_tier_to_h5(tier_a_frames, video_a, file_id, mode)
        write_tier_to_h5(tier_b_frames, video_b, file_id, mode)

        # remove the temporary files
        os.remove(f'{DATAPATH}/{file_id}.eaf')
        os.remove(f'{DATAPATH}/{file_id}.json.gz')


def write_tier_to_h5(tier, video, file_id, mode='train'):
    
    for tier_frame in tier:
        start, end, annotation = tier_frame
        poses = video[start:end]

        left_hand_keypoints = np.array([
            (pose['hand_left_keypoints_2d']).reshape(-1,3) for pose in poses
        ])
        right_hand_keypoints = np.array([
            (pose['hand_right_keypoints_2d']).reshape(-1,3) for pose in poses
        ]   )
        face_keypoints = np.array([
            (pose['face_keypoints_2d']).reshape(-1,3) for pose in poses
        ])
        pose_keypoints = np.array([
            (pose['pose_keypoints_2d']).reshape(-1,3) for pose in poses
        ])

        with h5py.File(f'{DATAPATH}/publicdgs_{mode}.h5', 'a') as f:
            # create a unique group for each clip
            group = f.create_group(f'{file_id}_{start}_{end}')
            group.create_dataset('left_hand_keypoints', data=left_hand_keypoints)
            group.create_dataset('right_hand_keypoints', data=right_hand_keypoints)
            group.create_dataset('face_keypoints', data=face_keypoints)
            group.create_dataset('pose_keypoints', data=pose_keypoints)
            group.create_dataset('text', data=annotation)


        