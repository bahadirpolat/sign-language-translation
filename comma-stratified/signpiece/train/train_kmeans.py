import os
import torch
from signpiece.quantizers.km import KMeansQuantizer
from signpiece.datasets import DatasetForQuantization, DatasetForQuantizationTraining


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Train kmeans model for sign language quantization')

    parser.add_argument('--save_dir', type=str, default='data', help='Path to save the quantized dataset' )
    parser.add_argument('--dataset_id', type=str, default='h2s', help='Dataset ID' )
    parser.add_argument('--dataset_dir', type=str, default='data', help='Path to the dataset directory' )
    parser.add_argument('--max_training_samples', type=int, default=None, help='Maximum number of instances to use for training' )
    parser.add_argument('--max_train_examples', type=int, default=None, help='Maximum number of examples to use for training' )
    parser.add_argument('--skip_frames', action='store_true', help='Skip frames')
    parser.add_argument('--codebook_size', type=int, default=512, help='Number of codes for quantization' )
    parser.add_argument('--pose_codes', type=int, default=512, help='Number of codes for pose quantization')
    parser.add_argument('--left_hand_codes', type=int, default=512, help='Number of codes for hand quantization')
    parser.add_argument('--right_hand_codes', type=int, default=512, help='Number of codes for hand quantization')
    parser.add_argument('--face_codes', type=int, default=512, help='Number of codes for face quantization')
    parser.add_argument('--dist_fn', type=str, default='l2', help='Distance function for kmeans' )
    parser.add_argument('--multicue', action='store_true', help='Use multicue data' )
    parser.add_argument('--transform', type=str, default=None, help='Transform to use for keypoints')
    parser.add_argument('--cues', type=list, default=['pose', 'left_hand', 'right_hand', 'face'], nargs='+', help='Cues to use for multicue data')
    parser.add_argument('--dev', action='store_true', help='Use dev mode')
    # OR Read from config file
    parser.add_argument('--config_file', type=str, default=None)

    return parser.parse_args()  

def load_from_config_file(args):
    # Update args from config file if provided
    import yaml
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            # Check if apply_bpe is set to True, if not exit program
            config = yaml.load(f, Loader=yaml.FullLoader)
            for key, value in config['quantization'].items():
                setattr(args, key, value)
            # add base args as well
            for key, value in config.items():
                if isinstance(value, dict): continue
                setattr(args, key, value)
    return args

def main(**kwargs):

    from argparse import Namespace
    kwargs = Namespace(**kwargs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(os.path.join(kwargs.save_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(kwargs.save_dir, 'models'), exist_ok=True)

    train = DatasetForQuantization(
        dataset_id=kwargs.dataset_id,
        mode='train' if not kwargs.dev else 'test',
        h5_fpath=kwargs.dataset_dir,
        transform=kwargs.transform,
        use_face=kwargs.use_face,
    )

    val = DatasetForQuantization(
        dataset_id=kwargs.dataset_id,
        mode='dev' if not kwargs.dev else 'test',
        h5_fpath=kwargs.dataset_dir,
        transform=kwargs.transform,
        use_face=kwargs.use_face,
    )

    test = DatasetForQuantization(
        dataset_id=kwargs.dataset_id,
        mode='test',
        h5_fpath=kwargs.dataset_dir,
        transform=kwargs.transform,
        use_face=kwargs.use_face,
    )

    print('Training KMeans model for {}...'.format(kwargs.dataset_id))
    
    kmeans = KMeansQuantizer(
        codebook_size=kwargs.codebook_size,
        dist_fn=kwargs.dist_fn,
        max_train_instances=kwargs.max_training_samples,
    )

    train_quantization = DatasetForQuantization(
        dataset_id=kwargs.dataset_id,
        mode='train' if not kwargs.dev else 'test',
        h5_fpath=kwargs.dataset_dir,
        transform=kwargs.transform,
        use_face=kwargs.use_face,
    )

    kmeans.fit(train_quantization)

    # Visualize the cluster centers
    save_path = os.path.join(kwargs.save_dir,)
    os.makedirs(save_path, exist_ok=True)

    kmeans.visualize_cluster_centers(save_path)

    # Infer the dataset
    kmeans.infer_dataset(train, os.path.join(kwargs.save_dir, 'data' ,f'{kwargs.dataset_id}.char.train.h5'), mode='train')
    kmeans.infer_dataset(val, os.path.join(kwargs.save_dir, 'data' ,f'{kwargs.dataset_id}.char.dev.h5'), mode='dev')
    kmeans.infer_dataset(test, os.path.join(kwargs.save_dir, 'data' ,f'{kwargs.dataset_id}.char.test.h5'), mode='test')

    # Save the kmeans model
    torch.save(kmeans, os.path.join(kwargs.save_dir,'models','kmeans_model.pth'))

if __name__ == '__main__':

    args = parse_args()
    args = load_from_config_file(args)

    main(**vars(args))
    
        