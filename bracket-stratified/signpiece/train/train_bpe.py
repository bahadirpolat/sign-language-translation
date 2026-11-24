import os
import h5py
from tqdm import tqdm
from tokenizers import Tokenizer

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    
    # Add tokenizer arguments
    parser.add_argument('--tokenizer', type=str, default=None, choices=['bpe', 'wordpiece'])
    parser.add_argument('--vocab_size', type=int, default=None, help='Vocabulary size')
    parser.add_argument('--min_frequency', type=int, default=None, help='Minimum frequency')
    parser.add_argument('--special_tokens', type=int, default=None, help='Special tokens')

    # Add augmentations
    parser.add_argument('--dataset_id', type=str, default='h2s')
    parser.add_argument('--num_initial_tokens', type=int, default=512)
    parser.add_argument('--multicue', action='store_true', help='Use multicue dataset')
    parser.add_argument('--augmentations', type=list, nargs='+', default=[])

    # Additional arguments
    parser.add_argument('--dataset_path', type=str, default='data/processed/processed.txt')
    parser.add_argument('--dev', action='store_true', help='Use dev split')

    # OR Read from config file
    parser.add_argument('--config_file', type=str, default=None)

    return parser.parse_args() 

def write_tokens_to_file(dataset, path, cue=None, augmentations=None):
    with open(path, 'w') as f:
        for data in tqdm(dataset, desc='Writing tokens to file'):
            str_tokens = '-'.join([str(i) for i in data['features']]) if not cue else '-'.join([str(i) for i in data[cue]]) 
            f.write(f'{str_tokens}\n')
            if augmentations:
                for aug in augmentations:
                    str_tokens = '-'.join([str(i) for i in data['features']]) if not cue else '-'.join([str(i) for i in data[cue]]) 
                    f.write(f'{str_tokens}\n')
    return path


def load_tokenizer(tokenizer, vocab_size, min_frequency):

    initial_vocab = [str(i) for i in range(args.num_initial_tokens)]

    if args.tokenizer == 'bpe':
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[PAD]", "[UNK]"]
        )

    if tokenizer == 'wordpiece':
        from tokenizers.models import WordPiece
        from tokenizers.trainers import WordPieceTrainer

        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[PAD]", "[UNK]"]
        )

    return tokenizer, trainer

def load_from_config_file(args):
    # Update args from config file if provided
    import yaml
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            print('Loading config file')
            config = yaml.load(f, Loader=yaml.FullLoader)
            # Check if apply_bpe is set to True, if not exit program
            if not config.get('apply_bpe'): 
                raise ValueError('apply_bpe is set to False in config file, skipping wordpiece training')
            for key, value in config['tokenization'].items():
                setattr(args, key, value)
            # add base args as well
            for key, value in config.items():
                if isinstance(value, dict): continue
                setattr(args, key, value)
    return args

if __name__ == '__main__':

    args = parse_args()
    args = load_from_config_file(args)

    os.makedirs(os.path.join(args.dataset_path, 'tokenizer'), exist_ok=True)

    if os.path.exists(
        os.path.join(args.dataset_path, 'data', f'{args.dataset_id}.bpe.train.h5')
    ) and os.path.exists(
        os.path.join(args.dataset_path, 'data', f'{args.dataset_id}.bpe.dev.h5')
    ) and os.path.exists(
        os.path.join(args.dataset_path, 'data', f'{args.dataset_id}.bpe.test.h5')
    ):
        raise ValueError('Tokenized files already exist, please remove them before running this script')

    if args.dataset_id == 'how2sign':
        args.dataset_id = 'h2s'

    from signpiece.datasets import DatasetForTokenization as Dataset
    train_dataset = Dataset(
        dataset_id=args.dataset_id,
        h5_fpath=os.path.join(args.dataset_path, 'data'),
        mode='train' if not args.dev else 'test', 
        skip_frames=False,
        multicue=args.multicue,
    )

    # write inference to file h5 file
    def infer(mode):
        dataset = Dataset(
            dataset_id=args.dataset_id,
            h5_fpath=os.path.join(args.dataset_path, 'data'),
            mode=mode if not args.dev else 'test', 
            skip_frames=False,
            multicue=args.multicue,
        )

        with h5py.File(f'{args.dataset_path}/data/{args.dataset_id}.bpe.{mode}.h5', 'a') as f:
            for data in tqdm(dataset, desc=f'Writing {mode} data to H5'):
                file_id, sentence = data['file_id'], data['sentence']
                if file_id not in f.keys():
                    f.create_group(file_id)
                    f.create_group(f'{file_id}/features')

                # Remove existing features
                if args.multicue:
                    if not (f[file_id]['features'].get(cue) is None): 
                        del f[f'{file_id}/features/{cue}']

                    # Encode initial tokens
                    init_tokens = data[cue]
                    str_tokens = '-'.join([str(i) for i in init_tokens])
                    encoded = tokenizer.encode(str_tokens).ids

                    f.create_dataset(f'{file_id}/features/{cue}', data=encoded)
                    if not 'sentence' in f[file_id].keys():
                        f.create_dataset(f'{file_id}/sentence', data=sentence)
                else:
                    if not (f[file_id].get('features') is None): 
                        del f[f'{file_id}/features']

                    # Encode initial tokens
                    init_tokens = data['features']
                    str_tokens = '-'.join([str(i) for i in init_tokens])
                    encoded = tokenizer.encode(str_tokens).ids

                    f.create_dataset(f'{file_id}/features', data=encoded)
                    if not 'sentence' in f[file_id].keys():
                        f.create_dataset(f'{file_id}/sentence', data=sentence)

    # Write initial tokens to file
    if args.multicue:
        for cue in [ 'pose', 'left_hand', 'right_hand', 'face']:
            print(f"Processing '{cue}' cue")
            # Load tokenizer
            tokenizer, trainer = load_tokenizer(args.tokenizer, args.vocab_size, args.min_frequency)
            # Write tokens to file
            tokens_path = write_tokens_to_file(
                train_dataset, f'{args.dataset_path}/tokenizer/{cue}.txt', cue=cue, augmentations=args.augmentations
            )
            # Train tokenizer
            tokenizer.train(
                [tokens_path],
                trainer=trainer
            )
            # Save tokenizer
            tokenizer.save(os.path.join(args.dataset_path, f'bpe.{cue}.json'))
            print(f"Tokenizer for '{cue}' cue saved to ", os.path.join(args.dataset_path, f'bpe.{cue}.json'))

            # Infer dataset(s)
            infer('train')
            infer('dev')
            infer('test')

    else:
        # Load tokenizer
        tokenizer, trainer = load_tokenizer(args.tokenizer, args.vocab_size, args.min_frequency)
        # Write tokens to file
        tokens_path = write_tokens_to_file(
            train_dataset, f'{args.dataset_path}/tokens.txt', cue=None, augmentations=args.augmentations
        )
        # Train tokenizer
        tokenizer.train(
            [tokens_path],
            trainer=trainer
        )
        # Save tokenizer
        tokenizer.save(os.path.join(args.dataset_path,'tokenizer','bpe.json'))
        print(f"Tokenizer saved to {os.path.join(args.dataset_path,'tokenizer','bpe.json')}")

        # Infer dataset(s)
        infer('train')
        infer('dev')
        infer('test')