import os
import math
import json
import torch
import pprint
import evaluate
import numpy as np
from transformers import (
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from tqdm.auto import tqdm
from signpiece.utils.hf import postprocess_text
from signpiece.utils.training_utils import (
    init_wandb,
    load_dataset,
    load_model,
    parse_args,
    load_from_config_file,
    format_dataset,
    evaluator
)

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":

    # Set the random seed
    torch.manual_seed(42)

    args = parse_args()
    args = load_from_config_file(args)

    init_wandb(args)

    pprint.pprint({
        "Model ID": args.model_id,
        "Model Name": args.model_name,
        "Model Type": args.model_type,
        "Dataset Type": args.dataset_id,
        "Objectives": args.objectives,
        "Use BPE":  args.use_bpe
    }, indent=4)

    sacrebleu = evaluate.load("sacrebleu")

    model, tokenizer, collate_fn, peft_params = load_model(args)

    # Load the dataset
    # Format the dataset
    train, tokenizer = load_dataset(args, tokenizer, mode="train" if not args.dev else "dev")
    train = format_dataset(args.model_type, train, tokenizer, args.max_translation_length, args.max_sign_length, model)
    
    val, tokenizer = load_dataset(args, tokenizer, mode="dev" if not args.dev else "dev")
    val = format_dataset(args.model_type, val, tokenizer, args.max_translation_length, args.max_sign_length, model)
    
    test, tokenizer = load_dataset(args, tokenizer, mode="test" if not args.dev else "dev")
    test = format_dataset(args.model_type, test, tokenizer, args.max_translation_length, args.max_sign_length, model)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
            preds = np.argmax(preds, axis=2)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        max_display = len(decoded_preds)
        for i in range(max_display):
            print(f"Prediction: {decoded_preds[i]}")
            print(f"Reference: {decoded_labels[i]}")
            print("*" * 50)

        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {
            "bleu": result["score"],
            "bleu-1": result["precisions"][0],
            "bleu-2": result["precisions"][1],
            "bleu-3": result["precisions"][2],
            "bleu-4": result["precisions"][3],
        }

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    print(
        f"Training size: {len(train)}",
        f"Validation size: {len(val)}",
        f"Test size: {len(test)}",
        f'Per device train batch size: {args.per_device_train_batch_size}',
        f'Gradient accumulation steps: {args.gradient_accumulation_steps}',
    )

    num_train_epochs = args.max_training_steps // (
        len(train)
        // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    num_train_epochs = max(math.ceil(num_train_epochs), 1)

    print(
        f" Model: {args.model_name}",
        f" Model ID: {args.model_id}",
        f" Dataset: {args.dataset_id}",
        f" Dataset ID: {args.dataset_id}",
        f" Training epochs: {num_train_epochs}",
        f" Number of training instances: {len(train)}",
        f" Number of validation instances: {len(val)}",
        f" Number of training steps: {args.max_training_steps}",
        f" Number of training examples: {len(train)}",
        f" Number of validation examples: {len(val)}",
        sep="\n",
    )

    from transformers import Seq2SeqTrainingArguments

    if not args.use_lm_tuner:
        optimizer = args.optimizer
        lr_scheduler = args.lr_scheduler_type
        learning_rate = float(args.learning_rate)
        weight_decay = float(args.weight_decay)

        training_args = Seq2SeqTrainer(
            # DIRECTORY
            output_dir=os.path.join(args.dataset_dir, "models", args.model_name),
            evaluation_strategy="steps",
            logging_steps=args.logging_steps,
            # BATCH SIZE
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # EVALUATION
            eval_accumulation_steps=args.eval_accumulation_steps,
            eval_steps=args.eval_steps,
            metric_for_best_model="bleu",
            generation_config=model.base_model.generation_config,
            predict_with_generate=True,
            # LEARNING RATE
            optim=optimizer,
            lr_scheduler_type=lr_scheduler,
            weight_decay=weight_decay,
            # OTHER
            save_strategy="no",
            save_total_limit=0,
            save_steps=args.save_steps,
            push_to_hub=args.push_to_hub,
            report_to=args.report_to if not args.dev else None,
            hub_model_id=f"{args.model_name}",
            hub_token=os.environ.get("HUB_TOKEN"),
            greater_is_better=True
        )

        if args.model_type == "t5-linear":
            from signpiece.utils.training_utils import collate_fn_linear
            collator = collate_fn_linear
        else:
            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            )

        # Pad the labels to the right
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        print("Training the model...")

        trainer.train()

        print("Model is trained exiting")
        # Unload device and run a forward pass on the test set
        trainer.model.eval()
    else:
        import pandas as pd
        from transformers import Seq2SeqTrainer, EvalPrediction

        if (
            args.model_type == "t5-linear"
        ):
            from signpiece.utils.training_utils import collate_fn_linear as collator
        else:
            collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            )

        def compute_metrics(eval_preds):

            if isinstance(eval_preds, tuple) and len(eval_preds) == 2:
                preds, labels = eval_preds
            elif isinstance(eval_preds, EvalPrediction): # qa uses
                preds, labels, _ = eval_preds.predictions, eval_preds.label_ids, eval_preds.inputs
            else:
                preds, labels, _ = eval_preds
            
            if isinstance(preds, tuple):
                preds = preds[0]

            # When predict_with_generate is False, we get predictions as a max of the logits
            if isinstance(preds, torch.Tensor) or isinstance(preds, np.ndarray):
                if preds.ndim == 3:
                    preds = np.argmax(preds, axis=2)

            # Replace -100s used for padding as we can't decode them
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            print("Computing metrics")
            for i in range(len(decoded_labels)):
                print(f"Prediction: {decoded_preds[i]}")
                print(f"Reference: {decoded_labels[i]}")
                print("*" * 50)

            results = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
            result = {
                "bleu": results["score"],
                "bleu-1": results["precisions"][0],
                "bleu-2": results["precisions"][1],
                "bleu-3": results["precisions"][2],
                "bleu-4": results["precisions"][3],
            }

            print("Results: ", result)

            return result

        optimizer = args.optimizer
        lr_scheduler = args.lr_scheduler_type
        learning_rate = float(args.learning_rate)
        weight_decay = float(args.weight_decay)

        training_args = Seq2SeqTrainingArguments(
            metric_for_best_model='eval_loss',
            load_best_model_at_end=True,
            # DIRECTORY
            output_dir=os.path.join(args.dataset_dir, "models", args.model_name),
            evaluation_strategy="steps",
            logging_steps=args.logging_steps,
            # BATCH SIZE
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # EVALUATION
            eval_accumulation_steps=args.eval_accumulation_steps,
            eval_steps=args.eval_steps,
            generation_config=model.model.generation_config,
            predict_with_generate=True,
            # LEARNING RATE
            optim=optimizer,
            lr_scheduler_type=lr_scheduler,
            learning_rate=learning_rate, # 5e-5 is the default works wonders in dev set
            # OTHER
            save_strategy="steps",
            save_total_limit=2,
            save_steps=args.save_steps,
            push_to_hub=args.push_to_hub,
            report_to=args.report_to if not args.dev else None,
            hub_model_id=f"{args.model_name}",
            hub_token=os.environ.get("HUB_TOKEN"),
            greater_is_better=True
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val,
            compute_metrics=compute_metrics,
            data_collator=collator,
        )

        trainer.train()
        results = trainer.evaluate(test)
        
        os.makedirs(args.output_dir, exist_ok=True)

        json.dump(results, open(os.path.join(args.output_dir, "results.json" ), "w"))

    # Save the model as pickle
    import pickle
    with open(os.path.join(args.output_dir, f"{args.model_name}.pkl"), "wb") as f:
        pickle.dump(trainer.model, f)

    def predict_and_evaluate(model, tokenizer, test, dataset_dir, mode="test"):

        eval_res = trainer.predict(test, metric_key_prefix=mode)

        # Save the evaluation results
        eval_predictions = []
        for (pred, label) in zip(eval_res.predictions, eval_res.label_ids):

            eval_predictions.append({
                "prediction": tokenizer.decode(pred, skip_special_tokens=False),
                "label": tokenizer.decode([l for l in label if l != -100 ], skip_special_tokens=True)
            })

        pprint.pprint(eval_predictions[:5])
        
        with open(os.path.join(args.output_dir, f"{mode}_results.json"), "w") as f:
            json.dump(eval_predictions, f)
        
        # Save the scores
        scores = eval_res.metrics
        with open(os.path.join(args.output_dir, f"{mode}_scores.json"), "w") as f:
            json.dump(scores, f)
    
    predict_and_evaluate(model, tokenizer, val, args.dataset_dir, 'dev')
    predict_and_evaluate(model, tokenizer, test, args.dataset_dir, 'test')
