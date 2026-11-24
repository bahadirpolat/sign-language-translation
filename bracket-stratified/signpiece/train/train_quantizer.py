import os
from tqdm.auto import tqdm
from .train_vq import main as main_vq
from .train_kmeans import main as main_kmeans
from .train_clip_vq import main as main_clip_vq

def parse_args():

    import argparse

    parser = argparse.ArgumentParser('Train CLIP-VQ model')

    # Run parameters
    parser.add_argument('--dev', action='store_true', help='Development mode')
    parser.add_argument('--dataset_id', type=str, required=True, help='Dataset ID')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the h5 file containing the dataset')
    parser.add_argument('--config_file', type=str, default=None, help='Path to the config file')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save the quantized dataset and model')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    
    # Dataset parameters

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

if __name__ == '__main__':
    
    args = parse_args()
    args = load_from_config_file(args)

    if os.path.exists(
        os.path.join(args.save_dir, 'data', f'{args.dataset_id}.char.train.h5')
    ) and os.path.exists(
        os.path.join(args.save_dir, 'data', f'{args.dataset_id}.char.dev.h5')
    ) and os.path.exists(
        os.path.join(args.save_dir, 'data', f'{args.dataset_id}.char.test.h5')
    ):
        print(f'Models for {args.dataset_id} already exist. Skipping training.')
        exit(0)

    if args.model_type == 'kmeans':
        main_kmeans(**vars(args))
    elif args.model_type == 'cvq':
        main_clip_vq(**vars(args))
    elif args.model_type == 'vqvae':
        main_vq(**vars(args))
    else:
        raise ValueError(f'Unknown model_type: {args.model_type}')