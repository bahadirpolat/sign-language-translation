import os
################### I have added an import - pprint
import pprint
###################
import math
import json
import re
################### for alignment *alg
import yaml
###################
import bert_score
import rouge
import torch
import wandb
import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)
from tqdm.auto import tqdm
from transformers import EvalPrediction
from signpiece.utils.hf import collate_fn
from signpiece.datasets import HFDatasetForSLT as DatasetForSLT

from dotenv import load_dotenv
load_dotenv()

def load_from_config_file(args, column_name="translation"):
    # Update args from config file if provided
    import yaml

    if args.config_file is not None:
        with open(args.config_file, "r") as f:
            print("Loading config file")
            config = yaml.load(f, Loader=yaml.FullLoader)
            for key, value in config[column_name].items():
                setattr(args, key, value)
            # add base args as well
            for key, value in config.items():
                if isinstance(value, dict):
                    continue
                setattr(args, key, value)
    return args

def init_wandb(args):
    if args.use_wandb:
        # Configure the wandb logger
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(
            project=args.project_name,
            name=f'{args.model_name}-{"-".join([obj["objective"] for obj in args.objectives])}-lr={args.learning_rate}-bs={args.per_device_train_batch_size}-steps={args.max_training_steps}-ctx={args.use_context}',
            tags=[
                args.model_id,
                ("char" if not args.use_bpe else "bpe"),
                ("multicue" if args.multicue else "singlecue"),
                ("skip_frames" if args.skip_frames else "full_frames"),
                ("lora" if args.lora else "baseline"),
            ]
            + [objective["objective"] for objective in args.objectives],
            entity="boun-pilab",
            config=vars(args),
        )

        return

    os.environ["WANDB_DISABLED"] = "true"

###################################################################### New function for task weights *alg
def get_task_weights_from_config(config_path):
    """
    Load task weights from config file.
    """
    if not config_path:
        # Return default if no config file is provided
        return {'sign2text': 1.0, 'alignment': 0.0}
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get task weights from config, with a fallback default
    task_weights = config.get('training', {}).get('task_weights', {
        'sign2text': 1.0,
        'alignment': 0.0
    })
    
    # Validate that weights sum to 1.0
    total_weight = sum(task_weights.values())
    if abs(total_weight - 1.0) > 0.001 and total_weight > 0:
        print(f"Warning: Task weights sum to {total_weight}, not 1.0. Normalizing...")
        task_weights = {k: v / total_weight for k, v in task_weights.items()}
    
    return task_weights
######################################################################
def load_model(args):
    # Initialize tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, token=os.environ.get("HUB_TOKEN")
    )

    if 'gpt' in args.model_id:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    from signpiece.signt5.modeling_signt5 import SignT5
    from signpiece.signt5.configuration_signt5 import SignT5Config

    if not args.continue_training:
        config = SignT5Config(
            base_model_name=args.model_id,
            sign_vocab_size=args.sign_vocab_size,
            sign_embed=args.sign_embed,
            embed_text=args.embed_text,
            learnable_sign_tokens=args.learnable_sign_tokens,
        )
        
        model = SignT5(config)
    else:
        model = SignT5.from_pretrained('karahansahin/'+args.model_name, )
    ######################################################## Add alignment tokens *alg
    print("\n" + "="*50)
    print("Adding task tokens for SLT and Alignment...")
    model.add_alignment_tokens(tokenizer)
    print("="*50 + "\n")
    ########################################################
    from transformers import GenerationConfig
    model.model.generation_config = GenerationConfig(
        max_length=args.max_translation_length,
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        do_sample=True,
        early_stopping=True,
        decoder_start_token_id=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    import torch
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn_linear(batch):
        # Embed text_input from model
        text_input = [ torch.tensor(item["text_inputs"]) for item in batch ]
        # Pad text_input to match max_token_length
        text_input = pad_sequence(text_input, batch_first=True, padding_value=tokenizer.pad_token_id)
        # remove the last token from the text_input
        text_input = text_input[:, :-1]
        text_attention_mask = torch.ones_like(text_input)

        sign_inputs = [ item["sign_inputs"] for item in batch ]
        # sign_inputs = [ torch.tensor(item["sign_inputs"]) for item in batch ]
        # Add padding to sign_inputs to match text_input shape
        sign_inputs = [
            # Add padding to sign_input to match text_input shape (seq_len, 255) -> (seq_len, model.config.d_model)
            torch.cat([
                item, 
                torch.zeros(item.shape[0], model.base_model_dim - item.shape[1])
            ], dim=1)
            for item in sign_inputs
        ]
        sign_input = pad_sequence(sign_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
        
        sign_mask = [ item["sign_attention_mask"] for item in batch ]
        sign_mask = pad_sequence(sign_mask, batch_first=True, padding_value=0)

        # Embed labels from model
        labels = [ torch.tensor(item["labels"]) for item in batch ]
        labels = pad_sequence(labels, padding_value=-100, batch_first=True)

        # Get paraphraphs undecoded
        # if batch[0].get('paraphrases'): paraphrases = [ item["paraphrases"] for item in batch ]
        # else: paraphrases = None

        return {
            "text_inputs": text_input,
            "sign_inputs": sign_input,
            "sign_attention_mask": sign_mask,
            "text_attention_mask": text_attention_mask,
            'labels': labels,
            # 'paraphrases': paraphrases 
        }

    return model, tokenizer, collate_fn_linear, None

def load_dataset(args, tokenizer, mode="train"):
    # This is a hack to concatenate two datasets together
    from torch.utils.data import ConcatDataset

    datasets = []
    for task in args.objectives:
        # Load the objective dataset
        objective, source_lang, target_lang = (
            task["objective"],
            task["source_lang"],
            task["target_lang"],
        )

        source_lang_code = "".join([word[0] for word in source_lang.split(" ")])

        # Update the tokenizer
        tokenizer.add_tokens(
            [f"<{source_lang_code}-{idx}>" for idx in range(args.sign_vocab_size)]
        )

        max_instances = args.max_val_samples if mode == "dev" else None
        if args.max_instances:
            max_instances = args.max_instances if args.dev else max_instances

        ################################################################### Task weights for SLT and ALG *alg
        # NOTE: task_weights are DEPRECATED - now using is_alignment_eligible flag from CSV
        # The flag-based task selection is handled in datasets.py __getitem__ method
        # for ALL modes (train/dev/test) consistently.
        task_weights = get_task_weights_from_config(args.config_file)
        
        print(f"=" * 80)
        print(f"Loading {mode.upper()} dataset - task selection via is_alignment_eligible flag")
        print(f"=" * 80)
        ################################################################### *alg
        objective_dataset = DatasetForSLT(
            dataset_id=args.dataset_id,
            annotation_file=args.annotation_file.replace("train", mode if not args.dev else "dev"),
            mode=mode if not args.dev else "dev",
            h5_fpath=os.path.join(args.dataset_dir),
            max_token_length=args.max_token_length,
            max_sign_length=args.max_sign_length,
            # max_translation_length=args.max_translation_length,
            skip_frames=args.skip_frames,
            tokenizer_name=args.model_id,
            objective=objective,
            source_lang=source_lang,
            target_lang=target_lang,
            num_codes=args.sign_vocab_size,
            model_type=args.model_type,
            use_bpe=args.use_bpe if args.model_type == "t5" else False,
            use_linear=True if args.model_type == "t5-linear" else False,
            use_char=True if args.model_type == "t5" else False,
            use_context=True if args.use_context else False,
            max_instances=max_instances,
            use_paraphrase=args.use_paraphrase,
            dataset_type=args.dataset_type, # main or long
            ###################################################################I modifies DatasetforSLT to include the weights *alg     
            task_weights=task_weights, 
            ###################################################################

        )

        if args.augmentations and mode == "train":

            from signpiece.datasets import AugmentedDatasetForSLT

            objective_dataset = AugmentedDatasetForSLT(
                objective_dataset,
                augmentations=args.augmentations,
            )

        print("*" * 50)
        if mode == "val" or mode == "test":
            return objective_dataset, tokenizer

        datasets.append(objective_dataset)

    from signpiece.datasets import MultiTaskDatasetForSLT

    dataset = MultiTaskDatasetForSLT(datasets)
    return dataset, tokenizer

def evaluator(model, tokenizer, dataset, args, mode="test"):

    def translator(_inputs):
        generated = model.generate(
            **_inputs,
            num_beams=args.num_beams,
            max_length=args.max_translation_length,
            early_stopping=args.early_stopping,
            bos_token_id=tokenizer.bos_token_id,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

        return tokenizer.batch_decode(generated, skip_special_tokens=True)
    
    results = []
    ########################################################################## *alg
    ##########################################################################
    for i in tqdm(range(len(dataset))):
        
        print(f"Example {i+1}/{len(dataset)}")
        
        if not args.model_type == "t5-linear":
            inputs = {'input_ids': dataset[i]["input_sequence"] }
        else: 
            inputs = { 
                'text_inputs': torch.tensor(dataset[i]["text_inputs"]).unsqueeze(0),
                'sign_inputs': torch.tensor(dataset[i]["sign_inputs"]).unsqueeze(0),
                'text_attention_mask': torch.tensor(dataset[i]["text_attention_mask"]).unsqueeze(0),
                'sign_attention_mask': torch.tensor(dataset[i]["sign_attention_mask"]).unsqueeze(0),
            }

            # Put the inputs on the right device
            _inputs = {k: v.to(model.device) for k, v in inputs.items()}
        ##################################################################################### Main ile çalışırken labels yerine paraphrases yaz
        labels = tokenizer.batch_decode([dataset[i]["paraphrases"]], skip_special_tokens=True)
        #####################################################################################
        translated_text = translator(_inputs)
        ####### !!!!!! Check this out later on.
        pprint.pprint({
            "prediction": translated_text[0],
            "original": labels
        })
        results.append({
            "prediction": translated_text[0],
            "original": labels
        })
    ###################################################################################################### 
    sacrebleu = evaluate.load("sacrebleu")
    result = sacrebleu.compute(
        predictions=[result["prediction"] for result in results],
        references=[result["original"] for result in results],
    )
    print({
        "bleu": result["score"],	
        "bleu-1": result["precisions"][0],
        "bleu-2": result["precisions"][1],
        "bleu-3": result["precisions"][2],
        "bleu-4": result["precisions"][3],
    })

    with open(os.path.join(args.dataset_dir, 'results', f"{mode}_inference.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # create a directory to save the results
    if not os.path.exists(os.path.join(args.dataset_dir, 'results')):
        os.makedirs(os.path.join(args.dataset_dir, 'results'))

    print(f"Evaluating the model on the {mode} set...")

    # Generate predictions
    test_results = model.predict(
        test_dataset=dataset,
        metric_key_prefix=mode,
    )

    eval_results = test_results.metrics

    # Save evaluation results
    with open(os.path.join(args.dataset_dir, 'results', f"{mode}_scores.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--model_name", type=str, default="h2s-test")
    parser.add_argument("--dataset_dir",type=str, default="data/How2Sign/")
    parser.add_argument("--sign_vocab_size", type=int, default=32000)
    ##################################################################################### I have changed the output directory
    # parser.add_argument("--output_dir", default="./results", type=str)
    parser.add_argument("--output_dir", default="/home/utkuo/SLT_output", type=str)
    #####################################################################################
    parser.add_argument("--project_name", type=str, default="tid-signpiece")
    parser.add_argument("--use_wandb", action="store_true")

    # Data processing
    parser.add_argument("--dataset_id", type=str, default="h2s")
    parser.add_argument("--dataset_type", type=str, default="main")
    parser.add_argument("--skip_frames", action="store_true")
    parser.add_argument("--max_token_length", type=int, default=128)
    parser.add_argument("--max_sign_length", type=int, default=250)
    parser.add_argument("--multicue", action="store_true")
    parser.add_argument("--use_bpe", action="store_true")
    parser.add_argument("--use_context", action="store_true")
    parser.add_argument("--use_paraphrase", action="store_true")
    parser.add_argument("--objectives", type=list, nargs="+", default=["sign2text"])
    parser.add_argument("--augmentations", type=list, nargs="+", default=None)
    #####################################################################################
    # The original annotation file loc: "data/How2Sign/annotations.json"
    parser.add_argument("--annotation_file", type=str, default="/ssd2/karahan/BUTID/sebeder.main.train.paraphrase.csv")
    #####################################################################################
    parser.add_argument("--max_translation_length", type=int, default=128)
    parser.add_argument("--learnable_sign_tokens", action="store_true")

    # Training arguments
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument("--model_id", type=str, default="t5-small")
    parser.add_argument("--max_training_steps", type=int, default=20_000)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=float, default=8)
    parser.add_argument("--eval_accumulation_steps", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")

    # Model arguments
    parser.add_argument("--sign_embed", type=str, default="linear")
    parser.add_argument("--embed_text", action="store_true")
    parser.add_argument("--use_shared", action="store_true")


    # Generation arguments
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=None)

    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_save", type=int, default=2)
    parser.add_argument("--max_instances", type=int, default=None)

    # LoRA arguments
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_bias", action="store_true")

    # Running arguments
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--config_file", type=str, default=None)

    return parser.parse_args()

######################################################################
# Timed BLEU and IoU Metrics for Alignment Task (Google Research Approach)
######################################################################

def parse_bracket_timestamp_format(text):
    """
    Parse bracket-delimited timestamp format into segments.
    Input: "[0.0] hello world [1.2] goodbye [2.5]"
    Output: [
        {start: 0.0, end: 1.2, text: "hello world"},
        {start: 1.2, end: 2.5, text: "goodbye"}
    ]
    """
    if not text or not isinstance(text, str):
        return []
    
    # Extract all timestamps using regex: [X.X] or [XX]
    timestamp_pattern = r'\[(\d+(?:\.\d+)?)\]'
    timestamps = re.findall(timestamp_pattern, text)
    
    if len(timestamps) < 2:
        return []  # Need at least 2 timestamps to form segments
    
    # Split text by timestamps to get the text between them
    text_parts = re.split(timestamp_pattern, text)
    
    segments = []
    
    # text_parts structure: ['', '0.0', ' hello world ', '1.2', ' goodbye ', '2.5', '']
    # We need text between each pair of timestamps
    for i in range(len(timestamps) - 1):
        start = float(timestamps[i])
        end = float(timestamps[i + 1])
        
        # Text is at position i*2 + 2 in text_parts
        # (assuming split pattern: pos 0=before first match, pos 1=first match, pos 2=between...)
        text_idx = i * 2 + 2
        if text_idx < len(text_parts):
            segment_text = text_parts[text_idx].strip()
            if segment_text:
                segments.append({
                    'start': start,
                    'end': end,
                    'text': segment_text
                })
    
    return segments


def unscale_bracket_timestamps(text, scale_factor=10):
    """
    Unscale timestamps in bracket format by dividing by scale_factor.
    Input: "[5] hello [12] world [27]"  (scaled ×10)
    Output: "[0.5] hello [1.2] world [2.7]"  (original 0.1 precision)
    
    This is applied BEFORE metric computation to compare fairly.
    """
    if not text or not isinstance(text, str):
        return text
    
    def unscale_match(match):
        val = float(match.group(1))
        unscaled = val / scale_factor
        return f'[{unscaled:.1f}]'
    
    # Match integer or float timestamps in brackets
    pattern = r'\[(\d+(?:\.\d+)?)\]'
    return re.sub(pattern, unscale_match, text)


def is_bracket_alignment_format(text):
    """
    Check if text is in bracket timestamp format.
    Returns True for: "[0.0] hello [1.2] world [2.5]"
    Returns False for regular SLT text.
    """
    if not text or not isinstance(text, str):
        return False
    # Check for bracket timestamp pattern at the start
    return bool(re.match(r'^\s*\[\d+(?:\.\d+)?\]', text))


# Keep old function for backward compatibility (comma format)
def parse_timestamp_format(text):
    """
    Parse comma-separated format into segments (legacy format).
    Input: "0,1,hello,1,2,world,2,3,bye"
    Output: [
        {start: 0, end: 1, text: "hello"},
        {start: 1, end: 2, text: "world"},
        {start: 2, end: 3, text: "bye"}
    ]
    """
    if not text or not isinstance(text, str):
        return []
    
    parts = text.split(',')
    segments = []
    
    i = 0
    while i + 2 < len(parts):
        try:
            start = float(parts[i].strip())
            end = float(parts[i + 1].strip())
            segment_text = parts[i + 2].strip()
            
            if segment_text:  # Only add non-empty text segments
                segments.append({
                    'start': start,
                    'end': end,
                    'text': segment_text
                })
            i += 3
        except (ValueError, IndexError):
            i += 1  # Skip malformed entries
    
    return segments


def assign_token_timestamps(segments):
    """
    Assign timestamps to each token using linear interpolation.
    Formula: token_time = start + (token_index × duration / num_tokens)
    """
    token_timestamps = []
    
    for segment in segments:
        start = segment['start']
        end = segment['end']
        text = segment['text']
        
        # Simple whitespace tokenization
        tokens = text.split()
        num_tokens = len(tokens)
        
        if num_tokens == 0:
            continue
            
        duration = end - start
        
        for idx, token in enumerate(tokens):
            token_time = start + (idx * duration / num_tokens)
            token_timestamps.append({
                'token': token,
                'time': token_time,
                'segment_start': start,
                'segment_end': end
            })
    
    return token_timestamps


def align_tokens_temporally(pred_tokens, ref_segments):
    """
    Match prediction tokens to reference segments based on temporal alignment.
    For each pred_token, find reference segment where:
        ref_segment.start <= pred_token.time < ref_segment.end
    """
    aligned_groups = {}  # key: ref_segment_id, value: {'pred_tokens': [], 'ref_text': str}
    
    for i, ref_seg in enumerate(ref_segments):
        aligned_groups[i] = {
            'pred_tokens': [],
            'ref_text': ref_seg['text']
        }
    
    for pred_token in pred_tokens:
        token_time = pred_token['time']
        
        # Find the reference segment this token belongs to
        for i, ref_seg in enumerate(ref_segments):
            if ref_seg['start'] <= token_time < ref_seg['end']:
                aligned_groups[i]['pred_tokens'].append(pred_token['token'])
                break
        else:
            # If no segment found, assign to the closest segment
            if ref_segments:
                # Find closest segment
                min_dist = float('inf')
                closest_idx = 0
                for i, ref_seg in enumerate(ref_segments):
                    dist = min(abs(token_time - ref_seg['start']), abs(token_time - ref_seg['end']))
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                aligned_groups[closest_idx]['pred_tokens'].append(pred_token['token'])
    
    return aligned_groups


def compute_timed_bleu(prediction, reference, use_bracket_format=True):
    """
    Compute Timed BLEU-2 for alignment task.
    
    1. Parse both prediction and reference (bracket or comma format)
    2. Assign token timestamps using linear interpolation
    3. Align prediction tokens to reference segments temporally
    4. Compute BLEU-2 for each aligned group
    5. Return mean BLEU-2 score
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    # Choose parser based on format
    if use_bracket_format:
        pred_segments = parse_bracket_timestamp_format(prediction)
        ref_segments = parse_bracket_timestamp_format(reference)
    else:
        pred_segments = parse_timestamp_format(prediction)
        ref_segments = parse_timestamp_format(reference)
    
    if not pred_segments or not ref_segments:
        return 0.0
    
    # Assign token timestamps to predictions
    pred_tokens = assign_token_timestamps(pred_segments)
    
    if not pred_tokens:
        return 0.0
    
    # Align prediction tokens to reference segments
    aligned_groups = align_tokens_temporally(pred_tokens, ref_segments)
    
    # Compute BLEU-2 for each aligned group
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for group_id, group in aligned_groups.items():
        pred_tokens_list = group['pred_tokens']
        ref_text = group['ref_text']
        ref_tokens = ref_text.split()
        
        if not pred_tokens_list or not ref_tokens:
            continue
        
        # BLEU-2: weights=(0.5, 0.5, 0, 0)
        try:
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens_list,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smoothing
            )
            bleu_scores.append(score)
        except Exception:
            continue
    
    if not bleu_scores:
        return 0.0
    
    return np.mean(bleu_scores) * 100


def compute_segment_iou(prediction, reference, use_bracket_format=True):
    """
    Compute IoU (Intersection over Union) at segment level.
    
    For each (pred_segment, gt_segment) pair:
        intersection = max(0, min(pred.end, gt.end) - max(pred.start, gt.start))
        union = max(pred.end, gt.end) - min(pred.start, gt.start)
        iou = intersection / union
    
    Returns: mean of all IoUs
    """
    # Choose parser based on format
    if use_bracket_format:
        pred_segments = parse_bracket_timestamp_format(prediction)
        ref_segments = parse_bracket_timestamp_format(reference)
    else:
        pred_segments = parse_timestamp_format(prediction)
        ref_segments = parse_timestamp_format(reference)
    
    if not pred_segments or not ref_segments:
        return 0.0
    
    ious = []
    
    # Match segments by order (simple 1:1 matching)
    for pred_seg, ref_seg in zip(pred_segments, ref_segments):
        pred_start, pred_end = pred_seg['start'], pred_seg['end']
        ref_start, ref_end = ref_seg['start'], ref_seg['end']
        
        intersection = max(0, min(pred_end, ref_end) - max(pred_start, ref_start))
        union = max(pred_end, ref_end) - min(pred_start, ref_start)
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    if not ious:
        return 0.0
    
    return np.mean(ious) * 100

######################################################################

def wrapper_for_compute_metrics(tokenizer):
    
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        bert_score = evaluate.load("bertscore")
        
        def compute_metrics(eval_preds):

            if isinstance(eval_preds, tuple) and len(eval_preds) == 2:
                preds, labels = eval_preds
            elif isinstance(eval_preds, EvalPrediction): # qa uses
                preds, labels = eval_preds.predictions, eval_preds.label_ids
                inputs = eval_preds.inputs
            else:
                preds, labels, inputs = eval_preds
            
            if isinstance(preds, tuple):
                preds = preds[0]

            # When predict_with_generate is False, we get predictions as a max of the logits
            if isinstance(preds, torch.Tensor) or isinstance(preds, np.ndarray):
                if preds.ndim == 3:
                    preds = np.argmax(preds, axis=2)

            # Decode labels without padding
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            if inputs:
                if inputs.get('paraphrases'):
                    # Don't need to decode, already have untokenized input
                    print(inputs['paraphrases'])
                    labels = inputs['paraphrases']
            else:
                # Replace -100s used for padding as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Separate alignment and SLT predictions based on format
            # Alignment format: "start,end,text,start,end,text,..."
            # SLT format: regular text without timestamp patterns
            alignment_preds = []
            alignment_labels = []
            slt_preds = []
            slt_labels = []
            
            for pred, label in zip(decoded_preds, decoded_labels):
                # Check if this is an alignment task output
                # Bracket format: "[0.0] text [1.2] text" OR scaled "[0] text [12] text"
                is_alignment = is_bracket_alignment_format(label)
                
                if is_alignment:
                    alignment_preds.append(pred)
                    alignment_labels.append(label)
                else:
                    slt_preds.append(pred)
                    slt_labels.append(label)
            
            print(f"\n{'='*60}")
            print(f"Computing metrics: {len(slt_preds)} SLT samples, {len(alignment_preds)} Alignment samples")
            print(f"{'='*60}\n")
            
            result = {}
            
            ################################################################
            # SLT Metrics (existing BLEU, ROUGE, BERT)
            ################################################################
            if slt_preds and slt_labels:
                print("Computing SLT metrics...")
                for i in range(min(5, len(slt_labels))):  # Print first 5
                    print(f"[SLT] Pred: {slt_preds[i][:80]}...")
                    print(f"[SLT] Ref:  {slt_labels[i][:80]}...")
                    print("-" * 40)
                
                bleu_res = bleu.compute(predictions=slt_preds, references=slt_labels)
                rouge_res = rouge.compute(predictions=slt_preds, references=slt_labels)
                bert_res = bert_score.compute(predictions=slt_preds, references=slt_labels, lang="tr")
                
                result.update({
                    "bleu-1": bleu_res["precisions"][0] * 100,
                    "bleu-2": bleu_res["precisions"][1] * 100,
                    "bleu-3": bleu_res["precisions"][2] * 100,
                    "bleu-4": bleu_res["precisions"][3] * 100,
                    "rouge1": rouge_res["rouge1"] * 100,
                    "rouge2": rouge_res["rouge2"] * 100,
                    "rougeL": rouge_res["rougeL"] * 100,
                    "rougeLsum": rouge_res["rougeLsum"] * 100,
                    "bert_score_precision": np.mean(bert_res["precision"]),
                    "bert_score_recall": np.mean(bert_res["recall"]),
                    "bert_score_f1": np.mean(bert_res["f1"]),
                })
            
            ################################################################
            # Alignment Metrics (Timed BLEU-2 and IoU)
            # Apply ÷10 unscaling before metric computation
            ################################################################
            if alignment_preds and alignment_labels:
                print("Computing Alignment metrics (Timed BLEU-2 and IoU)...")
                print("Note: Unscaling timestamps by ÷10 before evaluation")
                
                # Unscale timestamps (model outputs are ×10 scaled integers)
                unscaled_preds = [unscale_bracket_timestamps(p, scale_factor=10) for p in alignment_preds]
                unscaled_labels = [unscale_bracket_timestamps(l, scale_factor=10) for l in alignment_labels]
                
                for i in range(min(5, len(unscaled_labels))):  # Print first 5
                    print(f"[ALG] Pred (unscaled): {unscaled_preds[i][:80]}...")
                    print(f"[ALG] Ref  (unscaled): {unscaled_labels[i][:80]}...")
                    print("-" * 40)
                
                timed_bleu_scores = []
                iou_scores = []
                
                for pred, label in zip(unscaled_preds, unscaled_labels):
                    timed_bleu = compute_timed_bleu(pred, label, use_bracket_format=True)
                    iou = compute_segment_iou(pred, label, use_bracket_format=True)
                    timed_bleu_scores.append(timed_bleu)
                    iou_scores.append(iou)
                
                result.update({
                    "timed_bleu_2": np.mean(timed_bleu_scores) if timed_bleu_scores else 0.0,
                    "alignment_iou": np.mean(iou_scores) if iou_scores else 0.0,
                })
            
            # Set defaults for missing metrics
            if "bleu-4" not in result:
                result.update({
                    "bleu-1": 0.0, "bleu-2": 0.0, "bleu-3": 0.0, "bleu-4": 0.0,
                    "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0,
                    "bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0,
                })
            if "timed_bleu_2" not in result:
                result.update({
                    "timed_bleu_2": 0.0,
                    "alignment_iou": 0.0,
                })

            print("\n" + "="*60)
            print("Results:")
            for k, v in result.items():
                print(f"  {k}: {v:.4f}")
            print("="*60 + "\n")

            return result
        
        return compute_metrics