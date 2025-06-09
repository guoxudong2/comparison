import random
from utils_eval import get_metrics
from torchmetrics import AUROC
from Modules.NeuroGPT.src.neurogpt_backbone import NeuroGPTBackbone
import os, glob, gc, argparse, yaml, optuna
import numpy as np, torch, pytorch_lightning as pl, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from datasets.downstream.CHB_MIT_Scalp_EEG.chbmit_dataset import CHBMITDataset
from sklearn.model_selection import StratifiedShuffleSplit


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

class LitNeuroGPT(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        C, T = cfg["in_ch"], cfg["win_len"]
        self.model = NeuroGPTBackbone(
            n_channels=C,  # 23
            win_len=T,  # 500
            embed_dim=cfg["embed_dim"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            dropout=cfg["dropout"],
        )
        #self.crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg["cls_w"]))
        self.crit = torch.nn.CrossEntropyLoss()
        self.auroc = AUROC(task="binary")

        self.metrics = {"valid": [], "test": []}
        self.is_sanity = True  # 跳过第一次 sanity check
        self.best_thr = 0.5

        self.print_trainable_params()

    def forward(self, x):
        return self.model(x)
    def print_trainable_params(self):
        print("Trainable parameters in encoder/decoder:")
        for name, param in self.model.core.named_parameters():
            if param.requires_grad:
                print(f"{name} - {param.shape}")

    def _shared_step(self, batch, stage):
        x, y = batch
        logit = self(x)
        #print(f"logits: {logit[:5]}")
        loss = self.crit(logit, y)
        prob  = F.softmax(logit, -1)[:,1]
        #print(f"probs: {prob[:5]}")
        self.log(f"{stage}_loss", loss, prog_bar=True)
        if stage != "train":
            self.auroc.update(prob, y)
            self.metrics[stage].append((y.cpu(), prob.cpu()))
        return loss

    def training_step(self, b, _):  return self._shared_step(b, "train")
    def validation_step(self, b, _):self._shared_step(b, "valid")
    def on_validation_epoch_end(self):
        auc = self.auroc.compute()
        self.auroc.reset()
        self.log("valid_auc", auc, prog_bar=True)

        if not self.is_sanity:
            ys, ps = zip(*self.metrics["valid"])
            y_true = torch.cat(ys).numpy()
            y_prob = torch.cat(ps).numpy()

            names = ["accuracy", "balanced_accuracy", "precision",
                     "recall", "cohen_kappa", "f1", "roc_auc", "specificity"]
            res = get_metrics(y_prob, y_true, names,is_binary=True)
            for k, v in res.items():
                self.log(f"valid_{k}", v, prog_bar=(k == "roc_auc"))

            # 在验证集上搜索最佳阈值（以 balanced_accuracy 为主，其次 f1）
            from sklearn.metrics import balanced_accuracy_score, f1_score
            best_thr, best_bacc, best_f1 = 0.5, 0.0, 0.0
            for t in np.linspace(0.05, 0.95, 37):
                y_pred = (y_prob > t)
                bacc = balanced_accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                if (bacc > best_bacc) or (bacc == best_bacc and f1 > best_f1):
                    best_thr, best_bacc, best_f1 = t, bacc, f1
            self.best_thr = best_thr
            self.log("valid_thr_best", best_thr)
            self.log("valid_bacc_best", best_bacc)
            self.log("valid_f1_best", best_f1)

        # 重置
        self.metrics["valid"].clear()
        self.is_sanity = False

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        auc = self.auroc.compute()
        self.auroc.reset()
        self.log("test_auc", auc, prog_bar=True)

        # 用在 valid 上搜到的 self.best_thr 作为决策阈值
        ys, ps = zip(*self.metrics["test"])
        y_true = torch.cat(ys).numpy()
        y_prob = torch.cat(ps).numpy()
        names = ["accuracy", "balanced_accuracy", "precision",
                 "recall", "cohen_kappa", "f1", "roc_auc", "specificity"]
        res = get_metrics(y_prob, y_true, names,is_binary=True)
        for k, v in res.items():
            self.log(f"test_{k}", v, prog_bar=(k == "roc_auc"))

        # 3) 用 valid 上的最佳阈值评估 test 集
        y_pred = (y_prob > self.best_thr)
        from sklearn.metrics import balanced_accuracy_score, f1_score
        test_bacc = balanced_accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred)
        self.log("test_bacc_at_val_thr", test_bacc)
        self.log("test_f1_at_val_thr", test_f1)

        self.metrics["test"].clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        return {"optimizer": opt, "lr_scheduler": sch}

def load_yaml(p):
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def make_loaders(x_path, y_path, cfg_d):
    ds = CHBMITDataset(x_path, y_path, use_avg=True,
                       scale_div=10.0, interpolation_len=cfg_d["interpolation_len"])
    n_val = int(len(ds) * cfg_d["val_ratio"])
    tr_ds, va_ds = random_split(ds, [len(ds)-n_val, n_val])

    y_tr = np.array([ds.y[i] for i in tr_ds.indices])
    cls_w = compute_class_weight("balanced", classes=np.array([0,1]), y=y_tr)
    samp_w = [cls_w[int(l)] for l in y_tr]
    sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)

    tr_ld = DataLoader(tr_ds, batch_size=cfg_d["batch"], sampler=sampler, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=cfg_d["batch"], shuffle=False, num_workers=0)
    print("Sampler 1-batch label →", Counter(next(iter(tr_ld))[1].tolist()))
    return tr_ld, va_ld, cls_w.tolist()

# ───────── Optuna objective ─────────
def build_model(trial, cls_w, cfg_s, fixed, win_len):
    emb = fixed.get("embed_dim") or trial.suggest_categorical("embed_dim", cfg_s["embed_dim"])
    nly = fixed.get("n_layer")   or trial.suggest_categorical("n_layer",   cfg_s["n_layer"])
    nhe = fixed.get("n_head")    or trial.suggest_categorical("n_head",    cfg_s["n_head"])
    lr  = fixed.get("lr")        or trial.suggest_float("lr",
                        low=float(cfg_s["lr_low"]), high=float(cfg_s["lr_high"]), log=True)
    drp = fixed.get("dropout")   or trial.suggest_float("dropout",
                        low=cfg_s["dropout_low"], high=cfg_s["dropout_high"])

    cfg = dict(
        in_ch=23,
        win_len=win_len,
        embed_dim=emb,
        n_layer=nly,
        n_head=nhe,
        lr=lr,
        dropout=drp,
        cls_w=cls_w,
    )

    return LitNeuroGPT(cfg)

def objective(trial, tr_ld, va_ld, cls_w, cfg_s, cfg_t):
    model = build_model(trial, cls_w, cfg_s, fixed={}, win_len=cfg_d["interpolation_len"])
    early = EarlyStopping(monitor="valid_auc", mode="max",
                          patience=cfg_t["auc_patience"], verbose=False)
    trainer = pl.Trainer(max_epochs=cfg_t["max_epochs"],
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, logger=False, enable_checkpointing=False,
                         callbacks=[early], enable_progress_bar=False)
    trainer.fit(model, tr_ld, va_ld)
    return trainer.callback_metrics["valid_auc"].item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="../conf/neurogpt.yaml")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    global cfg_d, cfg_s, cfg_t
    cfg_d, cfg_s, cfg_t = cfg["data"], cfg["search"], cfg["train"]

    shift_dir = cfg_d["shift_dir"]
    x_files = sorted(glob.glob(os.path.join(shift_dir, "*_X_shift.npy")))
    assert x_files, f"No *_X_shift.npy in {shift_dir}"

    for x_path in x_files:
        pid = os.path.basename(x_path).split("_")[0]
        y_path = x_path.replace("_X_shift.npy", "_y_shift.npy")
        if not os.path.exists(y_path):
            continue
        print(f"\n======= Patient {pid} =======")

        tr_ld, va_ld, cls_w = make_loaders(x_path, y_path, cfg_d)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
        study.optimize(lambda t: objective(t, tr_ld, va_ld, cls_w, cfg_s, cfg_t),
                       n_trials=args.trials, timeout=args.timeout)
        best = study.best_params
        print("🏆 Best params:", best)

        # ---- final training ----
        best_model = build_model(None, cls_w, cfg_s, fixed=best,
                                 win_len=cfg_d["interpolation_len"])
        tb = pl_loggers.TensorBoardLogger("logs_neurogpt", "tb", pid)
        csv = pl_loggers.CSVLogger("logs_neurogpt", "csv", pid)
        lrmon = LearningRateMonitor(logging_interval="epoch")
        early = EarlyStopping(monitor="valid_auc", mode="max",
                              patience=cfg_t["auc_patience"], verbose=True)
        trainer = pl.Trainer(max_epochs=cfg_t["max_epochs"],
                             accelerator="gpu" if torch.cuda.is_available() else "cpu",
                             devices=1, logger=[tb, csv],
                             callbacks=[lrmon, early], log_every_n_steps=20,
                             enable_checkpointing=False)
        trainer.fit(best_model, tr_ld, va_ld)
        trainer.test(best_model, va_ld)

        del best_model, tr_ld, va_ld
        torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()