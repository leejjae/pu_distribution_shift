# PU Distribution Shift

This repository provides code for training models under positive-unlabeled (PU) learning with distribution shift.

## Installation

### Create a new environment (recommended)
```bash
# Using conda
conda create -n pu_shift python=3.9
conda activate pu_shift

# Or using venv
python -m venv pu_env
source pu_env/bin/activate  # On Windows: pu_env\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Verify installation
```bash
python -c "import torch; import torchvision; print('Installation successful!')"
```

## Dataset Preparation

> **Note:** Dataset paths are resolved relative to the repository root
> (e.g. `./data/cifar10v2/`, `./data/CIFAR-10-C/`). Always run the download
> scripts and training scripts from the repository root directory.

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

The download script names (`cifar10v2`, `cifar10c`) differ from the
`--test_dataset` argument values used during training. Use the following
values for `--test_dataset`:

| Downloaded dataset | `--test_dataset` value |
| ------------------ | ---------------------- |
| `cifar10v2`        | `cifarv2`              |
| `cifar10c`         | `cifar10c`             |

## Training Pipeline

> **Note:** Run all training scripts from the repository root. A CUDA-capable
> GPU is required.

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


**Example:**
```bash
python run_pretrain.py \
  --data ./data \
  --test_dataset cifarv2 \
  --train_prior 0.5 \
  --test_prior 0.5 \
  --epochs 500 \
  --batch-size 2048 \
  --checkpoint-dir ./checkpoint/
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


**Example:**
```bash
python run_self_training.py \
  --encoder ./checkpoint/cifar_cifarv2/seed42/cnn_cifar_train0.5_test0.5.pth \
  --test_dataset cifarv2 \
  --train_prior 0.5 \
  --test_prior 0.5 \
  --arch cnn_cifar \
  --epochs 200 \
  --batch_size 256 \
  --threshold 0.95 \
  --out_dir ./metric/st
```

This will save the evaluation results to:
```
./metric/st/<train_dataset>_<test_dataset>/seed<seed>/dst/train<train_prior>_test<test_prior>/<arch>/results.jsonl
```


## Output

- Pre-training checkpoints are saved in `./checkpoint/`
- Self-training results are saved in `./metric/st/` (or your specified `--out_dir`)
- Results include test accuracy, F1 score, and AUC metrics in JSON format