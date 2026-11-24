import os
import math
import json
import torch
import pprint
import evaluate
import numpy as np
from tqdm.auto import tqdm
from accelerate import Accelerator
# from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import HfApi
from transformers import get_scheduler, is_wandb_available
from signpiece.utils.hf import collate_fn, postprocess_text
from signpiece.utils.training_utils import (
    load_dataset,
    load_model,
    parse_args,
    load_from_config_file,
    format_dataset
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version


from dotenv import load_dotenv
load_dotenv()

def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)

def get_grad_norm(params, scale=1):
    """Compute grad norm.""" 
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm

# logger = get_logger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

if __name__ == "__main__":


    # Set the random seed
    torch.manual_seed(42)

    args = parse_args()
    args = load_from_config_file(args)

    send_example_telemetry("run_signpiece_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb' if args.use_wandb and is_wandb_available() else 'tensorboard',
        **accelerator_log_kwargs
    )

    import logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_main_process:

        # logger.info(accelerator.state, main_process_only=False)

        print(f"""
            Model ID: {args.model_id}
            Model Name: {args.model_name}
            Model Type: {args.model_type}
            Dataset Type: {args.dataset_id}
            Objectives: {args.objectives}
            Use BPE: {args.use_bpe}
        """)

        if args.use_wandb and is_wandb_available():
            import wandb

            wandb.init(
                project="tid-signpiece",
                entity="boun-pilab",
                name=f'{args.model_name}-{"-".join([obj["objective"] for obj in args.objectives])}-lr={args.learning_rate}-bs={args.per_device_train_batch_size}-steps={args.max_training_steps}-ctx={args.use_context}',
                config=vars(args),
            )

        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.model_name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(
                repo_name, 
                exist_ok=True, 
                token=os.environ.get('HUB_TOKEN')
            ).repo_id

            os.makedirs(args.output_dir, exist_ok=True)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")

        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    model, tokenizer, collate_fn_linear, peft_params = load_model(args)

    # Load the dataset
    # Format the dataset
    train, tokenizer = load_dataset(args, tokenizer, mode="train" if not args.dev else "dev")
    train = format_dataset(args.model_type, train, tokenizer, args.max_translation_length, args.max_sign_length, model)
    
    val, tokenizer = load_dataset(args, tokenizer, mode="dev" if not args.dev else "dev")
    val = format_dataset(args.model_type, val, tokenizer, args.max_translation_length, args.max_sign_length, model)
    
    test, tokenizer = load_dataset(args, tokenizer, mode="test" if not args.dev else "dev")
    test = format_dataset(args.model_type, test, tokenizer, args.max_translation_length, args.max_sign_length, model)

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

    # Initialize the data collator
    train_dataloader = torch.utils.data.DataLoader(
        train,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collate_fn_linear,
        shuffle=True,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        val,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collate_fn_linear,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collate_fn_linear,
    )
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.

    from transformers import Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=args.weight_decay,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_training_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_training_steps is None:
        args.max_training_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_training_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_training_steps
        if overrode_max_training_steps
        else args.max_training_steps * accelerator.num_processes,
    )

    metric = evaluate.load("sacrebleu")

    # # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_training_steps:
        args.max_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_training_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.save_steps

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # experiment_config = vars(args)
    # # TensorBoard cannot log Enums, need the raw value
    # accelerator.init_trackers("signt5_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    from transformers import EvalPrediction
    
    metric = evaluate.load("sacrebleu")

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

        results = metric.compute(predictions=decoded_preds, references=decoded_labels)
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

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

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
        greater_is_better=True,
        log_level="info",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
        data_collator=collate_fn_linear,
    )

    trainer.train()

    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train)}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    # logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_training_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_training_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)


    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if accelerator.state.num_processes > 1:
                loss = accelerator.gather_for_metrics(loss).sum()
                gradient_multiplied = accelerator.state.num_processes

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0 or step == len(train_dataloader) - 1:
                
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1.0
                )

                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)

                optimizer.step()
                optimizer.zero_grad()
                
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                loss.detach()

                train_logs = {
                    "train/loss": ( loss * args.gradient_accumulation_steps ).item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    'train/epoch': completed_steps / (num_update_steps_per_epoch * num_train_epochs),
                    'train/grad_norm': grad_norm,
                }
                
                if accelerator.is_local_main_process:
                    print({
                        k.split("/")[-1]: v for k, v in train_logs.items()
                    })
                    if  args.use_wandb:
                        wandb.log(train_logs, step=completed_steps)


            # This is used to save the model and tokenizer state every `checkpointing_steps`
            if isinstance(checkpointing_steps, int):
                if completed_steps+1 % checkpointing_steps == 0 and accelerator.sync_gradients:
                    # logger.info(f"Completed steps: {completed_steps} --Saving model to {args.output_dir}")
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)
                        api.upload_folder(
                            commit_message=f"Training in progress epoch {epoch}",
                            folder_path=args.output_dir,
                            repo_id=repo_id,
                            repo_type="model",
                            token=os.environ.get('HUB_TOKEN'),
                        )

            if completed_steps >= args.max_training_steps:
                break

        model.eval()

        eval_loss = 0
        
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                
                # Forward pass
                loss = model(**batch).loss

                eval_loss += loss.item()

                generated_tokens = accelerator.unwrap_model(model).generate(
                    sign_inputs=batch["sign_inputs"],
                    sign_attention_mask=batch["sign_attention_mask"],
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                print(f"BATCH {step}")
                for i in range(len(decoded_preds)):
                    print(f"Prediction: {decoded_preds[i]}")
                    print(f"Reference: {decoded_labels[i]}")
                    print("*" * 50)

                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )

        result = metric.compute()
        result = {
            "eval/loss": eval_loss / len(eval_dataloader),
            "eval/bleu": result["score"],
            "eval/bleu-1": result["precisions"][0],
            "eval/bleu-2": result["precisions"][1],
            "eval/bleu-3": result["precisions"][2],
            "eval/bleu-4": result["precisions"][3],
        }

        print(result)        

        wandb.log(result)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

            all_results = {f"eval_{k}": v for k, v in result.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)
