import math
import time
import json
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from lossFunc import bce_loss, nnpu_loss
from model import build_model
from dataset import DataManager, make_dataset
from utils_fixmatch import make_labels_from_fixmatch

# -------------------------
# Args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset", type=str, default="cifar")
parser.add_argument("--test_dataset", type=str, default="cifarv2", choices=["cifarv2","cifar10c","cinic"])
parser.add_argument("--data_root", type=Path, default=Path("./data"))
parser.add_argument("--train_prior", type=float, default=0.5)
parser.add_argument("--test_prior", type=float, default=0.5)

# model / ckpt
parser.add_argument("--arch", type=str, default='cnn_cifar')
parser.add_argument("--encoder", type=Path, required=True, help="pretrained encoder (state_dict .pth)")

# training
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--batch_size_val", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--val_split", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpu_id", type=int, default=0)

# DST / FixMatch
parser.add_argument("--threshold", type=float, default=0.95, help="FixMatch confidence")
parser.add_argument("--T", type=float, default=1.0, help="temperature for pseudo labeling")
parser.add_argument("--trade_off_self_training", type=float, default=1.0)
parser.add_argument("--freeze", action='store_true')
parser.add_argument("--trade_off_worst", type=float, default=1.0)
parser.add_argument("--mu", type=int, default=1, help="#unlabeled batches per labeled batch")

# logging
parser.add_argument("--out_dir", type=Path, default=Path("./metric/st"))
parser.add_argument("--tag", type=str, default=None)

# -------------------------
# Utils
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader, gpu, train_prior):
    model.eval()
    all_logits, all_labels, total_loss, total = [], [], 0.0, 0
    for x, y_true, y in loader:
        x = x.cuda(gpu)
        y_true = y_true.cuda(gpu)
        y = y.cuda(gpu)
        logits = model(x).squeeze(1)              

        y[y==0] = -1
        y_true[y_true==-1] = 0

        loss = nnpu_loss(logits, y, prior=train_prior)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y_true.detach().cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(np.int64)

    prob = 1 / (1 + np.exp(-all_logits))
    pred = (prob >= 0.5).astype(np.int64)

    acc = accuracy_score(all_labels, pred)
    f1  = f1_score(all_labels, pred, average="macro")
    try:
        auc = roc_auc_score(all_labels, prob)
    except ValueError:
        auc = float("nan")

    return (total_loss / max(1, total)), acc, f1, auc


class ConfidenceBasedSelfTrainingLoss(nn.Module):
    def __init__(self, tau=0.95, T=1.0):
        super().__init__()
        self.tau = tau
        self.T = T
    def forward(self, logits_strong, logits_weak):
        # hard target & mask from weak
        target, mask = make_labels_from_fixmatch(logits_weak, tau=self.tau, T=self.T)
        loss = F.binary_cross_entropy_with_logits(logits_strong, target, reduction="none")
        if mask.sum() > 0:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = logits_strong.new_tensor(0.)
        return loss, mask



class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, freeze: bool, linear_prob: bool):
        super().__init__()
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        feat_dim = backbone.feature_dim
        hid = max(32, feat_dim // 4)

        self.fc_pu = nn.Sequential(
            nn.Linear(feat_dim, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
            nn.Linear(hid,1)
        )

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x):
        z = self.forward_features(x)
        return self.fc_pu(z)

    def forward_all(self, x):
        z = self.forward_features(x)
        logit = self.fc_pu(z)
        logit_pseudo = self.fc_pu(z)
        return logit, logit_pseudo

# -------------------------
# Main
# -------------------------
def main():
    args = parser.parse_args()
    args.rank = 0
    set_seed(args.seed)

    # ----- Data -----
    dm = DataManager(
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        train_prior=args.train_prior,
        test_prior=args.test_prior
    )


    train_dataset, test_dataset = dm.get_data()

    sort_idx = np.argsort(test_dataset.y_true)
    te_4_tr = sort_idx[::2]   
    te_4_te = sort_idx[1::2]

    te_4_tr_dataset = Subset(test_dataset, te_4_tr)   
    te_4_te_dataset = Subset(test_dataset, te_4_te)   

    n_total = len(train_dataset)
    n_val = int(n_total*0.1)
    n_train = n_total-n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val], generator=generator)


    weak_train_x_dataset = make_dataset(train_dataset, role="weak_train")
    weak_val_x_dataset = make_dataset(val_dataset, role="weak_val")

    test_dataset    = make_dataset(te_4_te_dataset, role="test")

    tgt_x_dataset   = make_dataset(te_4_tr_dataset, role="tot_test")

    weak_train_x_loader = DataLoader(weak_train_x_dataset, batch_size=args.batch_size,
                                     shuffle=True, drop_last=True)
    weak_val_x_loader   = DataLoader(weak_val_x_dataset, batch_size=args.batch_size_val,
                                     shuffle=False, drop_last=False)
    test_loader         = DataLoader(test_dataset, batch_size=args.batch_size_val,
                                     shuffle=False, drop_last=False)
    tgt_loader     = DataLoader(tgt_x_dataset, batch_size=args.batch_size * args.mu,
                                     shuffle=True, drop_last=True)

    # ----- Model & Optim -----
    backbone = build_model(args.arch).cuda(args.gpu_id)
    state_dict = torch.load(args.encoder, map_location="cpu")
    backbone.load_state_dict(state_dict, strict=False)

    model = Classifier(backbone, freeze=args.freeze, linear_prob=True).cuda(args.gpu_id)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    st_criterion = ConfidenceBasedSelfTrainingLoss(tau=args.threshold, T=args.T)
    best_val_loss = math.inf
    best_state = None
    no_improve = 0
    t0 = time.time()

    # ----- Train -----
    unl_iter = iter(tgt_loader)
    for epoch in range(args.epochs):
        model.train()
        run_loss, N = 0.0, 0

        for (x_tr, y_true_tr, y_tr) in weak_train_x_loader:
            x_tr = x_tr.cuda(args.gpu_id)
            y_tr = y_tr.cuda(args.gpu_id)
            y_true_tr = y_true_tr.cuda(args.gpu_id)
            y_tr[y_tr==0] = -1

            try:
                (x_tgt_w, x_tgt_s), _, _ = next(unl_iter)
            except StopIteration:
                unl_iter = iter(tgt_loader)
                (x_tgt_w, x_tgt_s), _, _ = next(unl_iter)
            x_tgt_w = x_tgt_w.cuda(args.gpu_id)
            x_tgt_s = x_tgt_s.cuda(args.gpu_id)

            # concat → forward 3 heads (DST 원본 구조)
            x_cat = torch.cat([x_tr, x_tgt_w, x_tgt_s], dim=0)
            
            logit, logit_pseudo = model.forward_all(x_cat)

            B = x_tr.size(0)

            # 1) PU loss (labeled/source) on PU head
            tr_logit = logit[:B].squeeze(1)
            loss_pu = nnpu_loss(tr_logit, y_tr, prior=args.train_prior)

            # 2) Self-training loss (FixMatch): weak→strong on pseudo head
            tgt_w_logit, _ = logit[B:].chunk(2, dim=0)                  # weak on main head

            _, tgt_s_logit = logit_pseudo[B:].chunk(2, dim=0)           # strong on pseudo head
            
            self_training_loss, _ = st_criterion(tgt_s_logit.squeeze(1),
                                                 tgt_w_logit.squeeze(1))
            self_training_loss = args.trade_off_self_training * self_training_loss
            loss = loss_pu + self_training_loss
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * x_tr.size(0)
            N += x_tr.size(0)

        train_loss = run_loss / max(1, N)
        scheduler.step()

        # ----- validation on weak_val_x_loader (PU loss)
        model.eval()
        with torch.no_grad():
            val_loss, V = 0.0, 0
            for x, y_true, y in weak_val_x_loader:
                x = x.cuda(args.gpu_id)
                y = y.cuda(args.gpu_id)
                y[y==0] = -1
                logits = model(x).squeeze(1)
                loss = nnpu_loss(logits, y, prior=args.train_prior)
                val_loss += loss.item() * x.size(0)
                V += x.size(0)
            val_loss /= max(1, V)

        te_loss, te_acc, te_f1, te_auc = evaluate(model, test_loader, args.gpu_id, args.train_prior)

        print(f"[epoch {epoch:03d}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"acc={te_acc:.4f}  macroF1={te_f1:.4f}  auc={te_auc:.4f}",
              flush=True)

        # early stop
        if val_loss + 1e-9 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(json.dumps({"early_stop_at": epoch}), flush=True)
                break

    te_loss, te_acc, te_f1, te_auc = evaluate(model, test_loader, args.gpu_id, args.train_prior)

    # save
    pair_dir = f"{args.train_dataset}_{args.test_dataset}"
    seed_dir = f"seed{args.seed}"
    prior_dir = f"train{args.train_prior}_test{args.test_prior}"
    arch_dir = f"{args.arch}"
    root_out = args.out_dir / pair_dir / seed_dir / "dst" / prior_dir / arch_dir
    root_out.mkdir(parents=True, exist_ok=True)
    

    result = dict(train=args.train_dataset, test=args.test_dataset,
                  train_prior=args.train_prior, test_prior=args.test_prior,
                  arch=args.arch, seed=args.seed,
                  test_loss=te_loss, test_acc=te_acc, test_macro_f1=te_f1, test_auc=te_auc)
    with open(root_out / "results.jsonl", "w") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
