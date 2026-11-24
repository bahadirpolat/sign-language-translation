import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def collate_fn_hf(batch):
    # Add padding to the inputs
    # "inputs" must be 250 frames long
    # "attention_mask" must be 250 frames long
    # "labels" must be 128 tokens long
    return {
        "input_ids": torch.stack(
            [
                torch.cat(
                    (
                        sample["input_ids"].squeeze(0),
                        torch.zeros(512 - sample["input_ids"].shape[1]),
                    )
                )
                for sample in batch
            ]
        ).long(),
        "attention_mask": torch.stack(
            [
                (
                    torch.cat(
                        (
                            sample["attention_mask"],
                            torch.zeros(250 - sample["attention_mask"].shape[0]),
                        ),
                        dim=0,
                    )
                    if sample["attention_mask"].shape[0] < 250
                    else sample["attention_mask"]
                )
                for sample in batch
            ]
        ).long(),
        "labels": torch.stack(
            [
                (
                    torch.cat(
                        (
                            sample["labels"].squeeze(0),
                            torch.zeros(128 - sample["labels"].shape[0]),
                        ),
                        dim=0,
                    )
                    if sample["labels"].shape[0] < 128
                    else sample["labels"]
                )
                for sample in batch
            ]
        )
        .squeeze(0)
        .long(),
    }


def collate_fn(batch):
    # Add padding to the inputs
    # "inputs" must be 250 frames long
    # "attention_mask" must be 250 frames long
    # "labels" must be 128 tokens long
    return {
        "input_ids": torch.stack(
            [sample["input_ids"].squeeze(0) for sample in batch]
        ).long(),
        "sign_ids": torch.stack(
            [
                torch.cat(
                    (
                        sample["sign_ids"],
                        torch.zeros(
                            250
                            - batch[0]["input_ids"].size(
                                1
                            )  # NOTE: Max length of input_ids should be 250
                            - sample["sign_ids"].shape[0]
                        ),
                    ),
                    dim=0,
                )
                for sample in batch
            ]
        ).long(),
        "attention_mask": torch.stack(
            [
                (
                    torch.cat(
                        (
                            sample["attention_mask"],
                            torch.zeros(250 - sample["attention_mask"].shape[0]),
                        ),
                        dim=0,
                    )
                    if sample["attention_mask"].shape[0] < 250
                    else sample["attention_mask"]
                )
                for sample in batch
            ]
        ).long(),
        "labels": torch.stack(
            [
                (
                    torch.cat(
                        (
                            sample["labels"].squeeze(0),
                            torch.zeros(128 - sample["labels"].shape[0]),
                        ),
                        dim=0,
                    )
                    if sample["labels"].shape[0] < 128
                    else sample["labels"]
                )
                for sample in batch
            ]
        )
        .squeeze(0)
        .long(),
    }


# Add collate_fn to DataLoader
def collate_fn_multicue(batch):
    # Add padding to the inputs
    # "inputs" must be 250 frames long
    # "attention_mask" must be 250 frames long
    # "labels" must be 128 tokens long
    return {
        # NOTE: Same prompt for all cues if prompt augmentation is not used
        "input_ids": torch.stack(
            [sample["input_ids"].squeeze(0) for sample in batch]
        ).long(),
        "sign_ids": {
            f"{cue}": torch.stack(
                [
                    torch.cat(
                        (
                            sample["sign_ids"][cue].squeeze(0),
                            torch.zeros(
                                250
                                - batch[0]["input_ids"].size(
                                    1
                                )  # NOTE: Max length of input_ids should be 250
                                - sample["sign_ids"][cue].shape[1],
                                dtype=int,
                            ),
                        ),
                        dim=0,
                    )
                    for sample in batch
                ]
            ).long()
            for cue in ["pose", "left_hand", "right_hand", "face"]
        },
        "attention_mask": torch.stack(
            [
                (
                    torch.cat(
                        (
                            sample["attention_mask"],
                            torch.zeros(250 - sample["attention_mask"].shape[0]),
                        ),
                        dim=0,
                    )
                    if sample["attention_mask"].shape[0] < 250
                    else sample["attention_mask"]
                )
                for sample in batch
            ]
        ).long(),
        "labels": torch.stack(
            [
                (
                    torch.cat(
                        (
                            sample["labels"].squeeze(0),
                            torch.zeros(128 - sample["labels"].shape[0]),
                        ),
                        dim=0,
                    )
                    if sample["labels"].shape[0] < 128
                    else sample["labels"]
                )
                for sample in batch
            ]
        )
        .squeeze(0)
        .long(),
    }
