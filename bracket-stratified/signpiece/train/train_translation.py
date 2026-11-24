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
from transformers import Seq2SeqTrainingArguments
import pandas as pd
from signpiece.datasets import HFDatasetForSLT as DatasetForSLT
from signpiece.utils.training_utils import (
    init_wandb,
    load_dataset,
    load_model,
    parse_args,
    load_from_config_file,
    wrapper_for_compute_metrics
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

    model, tokenizer, collate_fn, peft_params = load_model(args)
    compute_metrics = wrapper_for_compute_metrics(tokenizer)

    # Load the dataset
    # Format the dataset
    train, tokenizer = load_dataset(args, tokenizer, mode="train" if not args.dev else "dev")    
    val, tokenizer = load_dataset(args, tokenizer, mode="dev" if not args.dev else "dev")
    test, tokenizer = load_dataset(args, tokenizer, mode="test" if not args.dev else "dev")
    
    
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

    optimizer = args.optimizer
    lr_scheduler = args.lr_scheduler_type
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)

    training_args = Seq2SeqTrainingArguments(
        metric_for_best_model='eval_bleu-4',
        load_best_model_at_end=True,
        # DIRECTORY

        ######################################## I have changed the output directory
        # output_dir=os.path.join(args.dataset_dir, "models", args.model_name),
        output_dir=args.output_dir,    # "/home/utkuo/SLT_output" previously 
        ########################################
        ######################################## I have changed the name evaluation_strategy to eval_strategy
        eval_strategy="steps",
        ########################################
        logging_steps=args.logging_steps,
        # BATCH SIZE
        max_steps=args.max_training_steps,
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
        # include_for_metrics=['paraphrases',] if args.use_paraphrase else [],
        log_level="info",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    trainer.train()
    results = trainer.evaluate(test)
    
    os.makedirs(args.output_dir, exist_ok=True)

    json.dump(results, open(os.path.join(args.output_dir, "results.json" ), "w"))

    # Save the model as pickle
    import pickle
    with open(os.path.join(args.output_dir, f"{args.model_name}.pkl"), "wb") as f:
        pickle.dump(trainer.model, f)

    def predict_and_evaluate(dataset, mode="test"):

        eval_res = trainer.predict(dataset, metric_key_prefix=mode)

        # Save the evaluation results
        eval_predictions = []
        for (pred, label) in zip(eval_res.predictions, eval_res.label_ids):

            eval_predictions.append({
                "prediction": tokenizer.decode(pred, skip_special_tokens=True),
                "label": tokenizer.decode([l for l in label if l != -100 ], skip_special_tokens=True)
            })

        pprint.pprint(eval_predictions[:5])
        
        with open(os.path.join(args.output_dir, f"{mode}_results.json"), "w") as f:
            json.dump(eval_predictions, f)
        
        # Save the scores
        scores = eval_res.metrics
        with open(os.path.join(args.output_dir, f"{mode}_scores.json"), "w") as f:
            json.dump(scores, f)
    
    predict_and_evaluate( val, 'dev' )
    predict_and_evaluate( test, 'test' )
