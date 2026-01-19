# PU Distribution Shift

This repository provides code for training models under positive-unlabeled (PU) learning with distribution shift.

## Installation
```bash
pip install -r requirements.txt
```

## Dataset Preparation

```bash
# Download CIFAR-10 v2
bash cifar_dataset.sh <cifar_dir> cifar10v2

# Download CIFAR-10-C
bash cifar_dataset.sh <cifar_dir> cifar10c
```

**Example:**
```bash
bash cifar_dataset.sh ./data cifar10v2
bash cifar_dataset.sh ./data cifar10c
```

## Training Pipeline

### Step 1: Pre-training with Barlow Twins

First, run the self-supervised pre-training script using Barlow Twins:

```bash
python run_pretrain.py \
  --data <data_path> \
  --train_prior <train_prior> \
  --test_dataset <test_dataset> \
  --train_prior <train_prior> \
  --test_prior <test_prior> \
  --epochs <num_epochs> \
  --batch-size <batch_size> \
  --checkpoint-dir <checkpoint_directory>
```


This will save the pre-trained encoder to:
```
./checkpoint/<train_dataset>_<test_dataset>/seed<seed>/<arch>_train<train_prior>_test<test_prior>.pth
```

### Step 2: Self-Training with Distribution Shift

After pre-training, run the self-training script using the pre-trained encoder:

```bash
python run_self_training.py \
  --encoder <path_to_pretrained_encoder> \
  --test_dataset <test_dataset> \
  --train_prior <train_prior> \
  --test_prior <test_prior> \
  --arch <architecture> \
  --epochs <num_epochs> \
  --batch_size <batch_size> \
  --threshold <confidence_threshold> \
  --out_dir <output_directory>
```


This will save the evaluation results to:
```
./metric/st/<train_dataset>_<test_dataset>/seed<seed>/dst/train<train_prior>_test<test_prior>/<arch>/results.jsonl
```


## Output

- Pre-training checkpoints are saved in `./checkpoint/`
- Self-training results are saved in `./metric/st/` (or your specified `--out_dir`)
- Results include test accuracy, F1 score, and AUC metrics in JSON format