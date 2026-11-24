# Alignment-Aware Turkish Sign Language Translation

EE491/EE492 Senior Design Project, Boğaziçi University. Team: Utku Özcan, Ahmet Bahadır Polat.

This project translates Turkish Sign Language (TİD) pose sequences into Turkish text **together with timestamps**. It doesn't only output *what* was signed, but also *when* each phrase was signed:

```
Video → "[0.0] Merhaba [1.2] nasılsın [2.8] bugün? [3.5]"
```

The model is SignT5 (`google/mt5-small` with a linear sign embedding). It is trained with multiple objectives: plain sign-to-text translation, and an alignment task that is used for clips marked `is_alignment_eligible` in the dataset.

## Repository layout

The repo has two experiment setups. Each one is a complete, runnable copy of the `signpiece` package, and they differ only in the timestamp format:

| Folder | Timestamp format | Precision | Example target |
|---|---|---|---|
| [`comma-stratified/`](comma-stratified) | `start,end,text,...` triplets | 1 s | `0,1,Merhaba,1,3,nasılsın` |
| [`bracket-stratified/`](bracket-stratified) | `[start] text [start] text ... [end]` | 0.1 s | `[0.0] Merhaba [1.2] nasılsın [2.8]` |

The bracket version multiplies timestamps by 10 before training, so `[1.2]` becomes `[12]` and the model learns integer tokens. It divides by 10 again before computing metrics. This lives in `datasets.py` and `utils/training_utils.py`.

```
├── bracket-stratified/
│   ├── signpiece/          # model, datasets, training code
│   ├── configs/config.yaml # training configuration
│   └── scripts/            # stratified train/dev/test re-split, tests
├── comma-stratified/       # same structure, comma format
├── docs/                   # EE491 & EE492 final reports, poster, presentation
├── requirements.txt
└── LICENSE
```

The main files we changed compared to the original signpiece package:
`signpiece/datasets.py`, `signpiece/signt5/modeling_signt5.py`, `signpiece/signt5/configuration_signt5.py`, `signpiece/train/train_translation.py`, `signpiece/utils/training_utils.py`, `signpiece/utils/hf.py`, `signpiece/utils/transforms.py`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add HF_TOKEN / WANDB_API_KEY if needed
```

## Data

The dataset (SEBEDER / BUTID pose data and annotation CSVs) is **not included** in this repository. To get access, contact the dataset owners at Boğaziçi University. After you have the data:

1. Create stratified splits (balanced by `is_alignment_eligible`): edit `INPUT_DIR` / `OUTPUT_DIR` in `scripts/stratified_resplit*.py`, then run it.
2. Set `dataset_dir` and `translation.annotation_file` in `configs/config.yaml`.

## Training

From inside one of the variant folders:

```bash
cd bracket-stratified
python -m signpiece.train.train_translation --config_file=configs/config.yaml
```

Note: `push_to_hub: True` in the config uploads checkpoints to the Hugging Face Hub. Set it to `False` if you don't want that.

## Results

See the [EE492 final report](docs/EE492_Final_Report.pdf) and [poster](docs/poster.pdf) for the evaluation (BLEU, ROUGE, timed BLEU and IoU) and a comparison of the two formats.

## Acknowledgements

This project builds on [signpiece](https://github.com/karahan-sahin/signpiece) by Karahan Şahin (MIT License, see [LICENSE](LICENSE)).
