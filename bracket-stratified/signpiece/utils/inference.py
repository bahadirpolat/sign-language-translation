import os
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from signpiece.datasets import HFDatasetForSLT
from signpiece.signt5.modeling_signt5 import SignT5

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model_id', type=str, default='signpiece/signt5-base-sebeder')
    parser.add_argument('--annotation_file', type=str, default='data/annotations/turkish_sign_language_annotations_train.json')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--dataset_dir', type=str, default='data/datasets/sebeder')
    parser.add_argument('--dataset_type', type=str, default='main')
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()

def inference():

    args = parse_args()

    model = SignT5.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model.config.base_model_name)

    print(f"Model loaded from {args.model_id}")
    val = HFDatasetForSLT(
        dataset_id='sebeder',
        annotation_file=args.annotation_file.replace('train', "dev"),
        mode="dev",
        h5_fpath=os.path.join(args.dataset_dir),
        max_token_length=256,
        max_sign_length=512,
        skip_frames=False,
        tokenizer_name=model.config.base_model_name,
        objective='sign2text',
        source_lang='Türk İşaret Dili',
        target_lang='Türkçe',
        num_codes=100,
        model_type='t5-linear',
        use_bpe=False,
        use_linear=True,
        use_char=False,
        use_context=False,
        use_paraphrase=False,
        dataset_type=args.dataset_type,
        max_instances=20 if args.debug else None,
    )

    test = HFDatasetForSLT(
        dataset_id='sebeder',
        annotation_file=args.annotation_file.replace('train', "test"),
        mode="test",
        h5_fpath=os.path.join(args.dataset_dir),
        max_token_length=256,
        max_sign_length=512,
        skip_frames=False,
        tokenizer_name=model.config.base_model_name,
        objective='sign2text',
        source_lang='Türk İşaret Dili',
        target_lang='Türkçe',
        num_codes=100,
        model_type='t5-linear',
        use_bpe=False,
        use_linear=True,
        use_char=False,
        use_context=False,
        use_paraphrase=False,
        dataset_type=args.dataset_type,
        max_instances=20 if args.debug else None,
    )

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

        labels = [ torch.tensor(item["labels"]) for item in batch ]
        if batch[0].get('paraphrases') is not None:
            labels = [ [
                [p] for p in item["paraphrases"]
            ] for item in batch ]
        else:
            # Decode the labels to sentences
            labels = [ [tokenizer.decode(item, skip_special_tokens=True)] for item in labels ]

        return {
            "file_ids": [ item["file_id"] for item in batch ],
            "text_inputs": text_input,
            "sign_inputs": sign_input,
            "sign_attention_mask": sign_mask,
            "text_attention_mask": text_attention_mask,
            'labels': labels
        }
    
    val_loader = DataLoader(
        val,
        batch_size=args.batch_size,
        collate_fn=collate_fn_linear,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test,
        batch_size=args.batch_size,
        collate_fn=collate_fn_linear,
        shuffle=False,
        num_workers=4
    )

    print(f"Test dataset size: {len(test)}")
    print(f"Validation dataset size: {len(val)}")

    sample = next(iter(val_loader))

    print(f"Validation dataset size: {len(val)}")

    def inference_on_loader(loader, split):

        clip_ids, predictions, references = [], [], []

        model.eval()
        for batch in tqdm(loader, desc=f"Inference on {split}", total=len(loader)):
            text_inputs = batch['text_inputs']
            sign_inputs = batch['sign_inputs']
            sign_attention_mask = batch['sign_attention_mask']
            text_attention_mask = batch['text_attention_mask']
            labels = batch['labels']

            outputs = model.generate(
                text_inputs=text_inputs,
                text_attention_mask=text_attention_mask,
                sign_inputs=sign_inputs,
                sign_attention_mask=sign_attention_mask,
                # decoder_attention_mask=None,
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

            # Decode the outputs
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_outputs)
            references.extend(labels)
            clip_ids.extend(batch['file_ids'])

        return clip_ids, predictions, references
    
    # Calculate BLEU scores
    from evaluate import load
    bleu_metric = load('sacrebleu')
    # wer_metric = load('wer')
    # rouge_metric = load('rouge')

    val_ids, val_predictions, val_references = inference_on_loader(val_loader, 'val')


    val_bleu = bleu_metric.compute(
        predictions=val_predictions,
        references=val_references,
    )
    # val_wer = wer_metric.compute(
    #     predictions=val_predictions,
    #     references=val_references,
    # # )
    # val_rouge = rouge_metric.compute(
    #     predictions=val_predictions,
    #     references=val_references,
    # )

    test_ids, test_predictions, test_references = inference_on_loader(test_loader, 'test')
    
    test_bleu = bleu_metric.compute(
        predictions=test_predictions,
        references=test_references,
    )
    # test_wer = wer_metric.compute(
    #     predictions=test_predictions,
    #     references=test_references,
    # )
    # test_rouge = rouge_metric.compute(
    #     predictions=test_predictions,
    #     references=test_references,
    # )

    # Save the predictions and BLEU scores

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    import json
    with open(os.path.join(args.output_dir, 'val_inference.json'), 'w') as f:
        json.dump([
            {"clip_id": val_ids[i], 'predictions': val_predictions[i], 'references': val_references[i]}
            for i in range(len(val_predictions))
        ], f, indent=4)
    
    with open(os.path.join(args.output_dir, 'test_inference.json'), 'w') as f:
        json.dump([
            {"clip_id": test_ids[i], 'predictions': test_predictions[i], 'references': test_references[i]}
            for i in range(len(test_predictions))
        ], f, indent=4)

    with open(os.path.join(args.output_dir, 'val_bleu.json'), 'w') as f:
        json.dump({
            "bleu": val_bleu["score"],
            "bleu-1": val_bleu["precisions"][0],
            "bleu-2": val_bleu["precisions"][1],
            "bleu-3": val_bleu["precisions"][2],
            "bleu-4": val_bleu["precisions"][3],
        }, f, indent=4)

    with open(os.path.join(args.output_dir, 'test_bleu.json'), 'w') as f:
        json.dump({
            "bleu": test_bleu["score"],
            "bleu-1": test_bleu["precisions"][0],
            "bleu-2": test_bleu["precisions"][1],
            "bleu-3": test_bleu["precisions"][2],
            "bleu-4": test_bleu["precisions"][3],
        }, f, indent=4)

if __name__ == '__main__':

    inference()