
# This script is used to transform the Phoenix2014T dataset into a csv format
# https://github.com/neccam/slt/blob/master/base_annotations/phoenix14t.{mode}
import os
import gzip
import h5py
import pickle
import pandas as pd

def load_annotations(data_path='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/Phoenix/', mode='train'):

    dpath = os.path.join(data_path, 'phoenix14t.{}'.format(mode))
    with gzip.open(dpath.format('train'), 'rb') as f:
        loaded_object = pickle.load(f)

    df = pd.DataFrame(loaded_object)
    df.drop(columns=['alignments'], inplace=True)
    df.to_csv(dpath.format(f'.{mode}.csv'))

    return df


def parse_args():
    
    import argparse

    parser = argparse.ArgumentParser(description="Transform Phoenix2014T dataset into a csv format")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/Phoenix/",
        help="Path to Phoenix2014T dataset"
    )
    parser.add_argument(
        "--phoenix_pose",
        type=str,
        default="/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/Phoenix/pose_phoenix2014T_up_hrnet_TxN_wh.pkl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/",
        help="Path to save the csv files"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    phoenix_pose = pickle.load(open(args.phoenix_pose, 'rb'))

    for mode in ['train', 'dev', 'test']:
        annotations = load_annotations(data_path=args, mode=mode)
        h5_file = os.path.join(args.output_path, f'phoenix14t.{mode}.h5')
        h5_file = h5py.File(h5_file, 'a')   
        for annotation in annotations:
            
