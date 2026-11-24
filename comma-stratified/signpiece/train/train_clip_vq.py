import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from signpiece.datasets import DatasetForContrastiveTraining

from signpiece.quantizers.cvq import CLVQ, collate_fn

from dotenv import load_dotenv
load_dotenv()

import argparse

def parse_args():

    parser = argparse.ArgumentParser('Train CLIP-VQ model')

    # Config file
    parser.add_argument('--config_file', type=str, default=None, help='Path to the config file')

    # Run parameters
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the h5 file containing the dataset')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save the quantized dataset and model')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--dev', action='store_true', help='Development mode')
    
    # Dataset parameters
    parser.add_argument('--dataset_id', type=str, required=True, help='Dataset ID')
    parser.add_argument('--train_split_percentage', type=float, default=0.9, help='Split to use for training')
    parser.add_argument('--max_sequence_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--min_sequence_length', type=int, default=10, help='Minimum sequence length')

    # Model parameters
    parser.add_argument('--input_dim', type=int, default=144, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')

    # VQ parameters
    parser.add_argument('--vq_type', type=str, default='vq-ste++', help='Type of vector quantization to use')
    parser.add_argument('--commitment_cost', type=float, default=1, help='Commitment cost')
    parser.add_argument('--rotation_trick', action='store_true', help='Use rotation trick')
    parser.add_argument('--codebook_size', type=int, default=1024, help='Codebook size')
    parser.add_argument('--freeze_codebook', action='store_true', help='Freeze the codebook')
    parser.add_argument('--commit_coef', type=float, default=0.25, help='Commitment loss coefficient')
    parser.add_argument('--post_linear', action='store_true', help='Use post linear layer')

    # Inference parameters
    parser.add_argument('--window_size', type=list, default=[20], nargs='+', help='Window size for inference')
    parser.add_argument('--stride', type=list, default=[5], nargs='+', help='Stride for inference')

    # Loss parameters
    parser.add_argument('--margin', type=float, default=1, help='Margin for triplet loss')
    parser.add_argument('--loss_type', type=str, default='cosine', help='Type of loss to use')
    parser.add_argument('--dist_fn', type=str, default='euclidean', help='Distance metric to use for triplet loss')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_training_samples', type=int, default=1000, help='Maximum number of samples to use for training')
    parser.add_argument('--max_training_epochs', type=int, default=100, help='Maximum number of epochs to train the model')
    
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

    # Change kwargs into namespace
    kwargs = argparse.Namespace(**kwargs)

    import os
    # Add cuda blocking to avoid memory fragmentation
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Parse command line arguments
    kwargs.run_name = f'{kwargs.dataset_id}-{kwargs.vq_type}-l={kwargs.loss_type}-{kwargs.dist_fn[:3]}-{kwargs.codebook_size}-{kwargs.commitment_cost}-{kwargs.commit_coef}-{kwargs.max_training_samples}-{float(kwargs.learning_rate)}'

    os.makedirs(os.path.join(kwargs.save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(kwargs.save_dir, 'data'), exist_ok=True)

    # Initialize the Trainer and pass the wandb_logger
    if kwargs.use_wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb_logger = WandbLogger(
            name=kwargs.run_name,
            project='tid-quantizer',
            entity='boun-pilab',
            tags=[
                'clip-vq',
                kwargs.dataset_id,
                'vanilla-vq' if kwargs.vq_type == 'vanilla' else 'vq-ste++',
                'dist-fn-' + kwargs.dist_fn[:3],
                'loss-' + kwargs.loss_type,
            ],
            config=vars(kwargs)
        )
    else: 
        wandb_logger = None

    dataset = DatasetForContrastiveTraining(
        dataset_id=kwargs.dataset_id,
        h5_fpath=kwargs.dataset_dir,
        mode='train' if not kwargs.dev else 'test',
        max_instances=kwargs.max_training_samples,
        transform='yasl',
        min_sequence_length=kwargs.min_sequence_length,
        max_sequence_length=kwargs.max_sequence_length,
    )

    train, val = torch.utils.data.random_split(dataset, [int(kwargs.train_split_percentage * len(dataset)), len(dataset) - int(kwargs.train_split_percentage * len(dataset))])

    train_loader = DataLoader(train,
                              batch_size=kwargs.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True)
    
    val_loader = DataLoader(val,
                            batch_size=kwargs.batch_size,
                            collate_fn=collate_fn,
                            shuffle=False)

    # Initialize the model
    model = CLVQ(
        input_dim=kwargs.input_dim,
        hidden_dim=kwargs.hidden_dim,
        num_layers=kwargs.num_layers,
        num_heads=kwargs.num_heads,
        codebook_size=kwargs.codebook_size,
        commitment_cost=kwargs.commitment_cost,
        num_training_samples=kwargs.max_training_samples,
        batch_size=kwargs.batch_size,
        learning_rate=float(kwargs.learning_rate),
        commit_coef=kwargs.commit_coef,
        use_wandb=kwargs.use_wandb,
        loss_type=kwargs.loss_type,
        dist_fn=kwargs.dist_fn,
        margin=kwargs.margin,
        vq_type=kwargs.vq_type,
    )

    # Move the model to GPU if available
    model = model.cuda() if torch.cuda.is_available() else model

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=kwargs.max_training_epochs,
                         accelerator='auto',
                         accumulate_grad_batches=kwargs.gradient_accumulation_steps,
                         log_every_n_steps=1,
                         default_root_dir=os.path.join(kwargs.save_dir, 'models'),
                         logger=wandb_logger)

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    print("Training completed!")

    model = model.cuda() if torch.cuda.is_available() else model
    # Infer all indices
    print("Infering all indices...")
    for mode in ['train', 'dev', 'test']:

        print(f"Mode: {mode}")

        dataset = DatasetForContrastiveTraining(
            h5_fpath=kwargs.dataset_dir,
            dataset_id=kwargs.dataset_id,
            mode=mode if not kwargs.dev else 'test',
            transform='yasl',
            get_origin=True
        )

        model.infer_dataset(dataset, 
                            mode=mode,
                            windows=kwargs.window_size, 
                            strides=kwargs.stride,
                            save_dir=os.path.join(kwargs.save_dir, 'data' ,f'{kwargs.dataset_id}.char.{mode}.h5'))

    print("Inference completed!")

if __name__ == "__main__":

    kwargs = parse_args()
    load_from_config_file(kwargs)
    main(**vars(kwargs))