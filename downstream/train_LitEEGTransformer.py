'''import random
import os
import numpy as np
import yaml, argparse, torch
from dotmap import DotMap
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from datasets.downstream.CHB_MIT_Scalp_EEG.chbmit_dataset import CHBMITDataset
from Modules.LitTransformer.LitEEGTransformer import LitEEGTransformer

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(7)

# ---------- YAML + CLI ----------
def load_cfg(yaml_path, overrides):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = DotMap(yaml.safe_load(f))
    for item in overrides:
        key, val = item.split('=', 1)
        node = cfg
        *ks, last = key.lstrip('-').split('.')
        for k in ks:
            node = node[k]
        # auto cast
        if val.lower() in ['true','false']:
            val = val.lower() == 'true'
        else:
            try: val = int(val)
            except:
                try: val = float(val)
                except: pass
        node[last] = val
        cfg.train.lr = float(cfg.train.lr)
    return cfg

# ---------- Data ----------
def build_ld(cfg):
    ds = CHBMITDataset(cfg.data.x_path, cfg.data.y_path,
                       use_avg=True, scale_div=10.0,
                       interpolation_len=cfg.data.interpolation_len)

    print('Dataset 标签分布:',np.bincount(ds.y))
    n_val = int(len(ds)*cfg.data.val_ratio)
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    labels_train = [ds.y[i].item() for i in train_ds.indices]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)

    labels = [ds.y[i] for i in train_ds.indices]
    pos, neg = sum(labels), len(labels)-sum(labels)
    w = [1/pos if l==1 else 1/neg for l in labels]
    sampler = WeightedRandomSampler(w, len(w), replacement=True)

    tr_ld = DataLoader(train_ds, batch_size=cfg.data.batch, sampler=sampler, num_workers=0, persistent_workers=False)
    #tr_ld = DataLoader(train_ds, batch_size=cfg.data.batch, shuffle=True, num_workers=0, persistent_workers=False)
    va_ld = DataLoader(val_ds,   batch_size=cfg.data.batch, shuffle=False, num_workers=0, persistent_workers=False)
    return tr_ld, va_ld, class_weights

# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='../conf/liteegtransformer.yaml',
                        help='YAML config file')
    parser.add_argument('opts', nargs=argparse.REMAINDER,
                        help='Override cfg e.g. model.d_model=256 train.lr=1e-4')
    args = parser.parse_args()

    cfg = load_cfg(args.cfg, args.opts)
    tr_ld, va_ld, class_weights = build_ld(cfg)

    model = LitEEGTransformer(lr=float(cfg.train.lr),
                         in_ch=cfg.model.in_ch,
                         out_ch=cfg.model.out_ch,
                         T=cfg.model.T,
                         class_weights=class_weights)

    logger = pl_loggers.TensorBoardLogger(
        './logs_liteegtransformer/', name='chb01')

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[cfg.train.gpu] if torch.cuda.is_available() else 1,
        logger=logger,
        log_every_n_steps=10
    )
    trainer.fit(model, tr_ld, va_ld)
    trainer.test(model, va_ld)'''

# train_shifted_all.py
import os, glob, yaml, argparse, random
import numpy as np, torch, pytorch_lightning as pl
from dotmap import DotMap
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from pytorch_lightning import loggers as pl_loggers

from datasets.downstream.CHB_MIT_Scalp_EEG.chbmit_dataset import CHBMITDataset
from Modules.LitTransformer.LitEEGTransformer import LitEEGTransformer


# reproducibility ------------------------------------------------------------
def seed_all(seed=1029):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
seed_all(7)

# load only non-data hyper-params -------------------------------------------
def load_cfg(tmpl: str) -> DotMap:
    with open(tmpl, 'r', encoding='utf-8') as f:
        cfg = DotMap(yaml.safe_load(f))
    # 删除 data 路径字段（防止误用）
    cfg.data.pop('x_path', None)
    cfg.data.pop('y_path', None)
    return cfg

# build loaders for ONE patient --------------------------------------------
def make_loaders(x_path, y_path, cfg):
    ds = CHBMITDataset(x_path, y_path,
                       use_avg=True, scale_div=10.0,
                       interpolation_len=cfg.data.interpolation_len)

    n_val = int(len(ds) * cfg.data.val_ratio)
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val])

    y_tr = np.array([ds.y[i] for i in tr_ds.indices])
    cls_w = compute_class_weight(class_weight='balanced', classes=np.unique(y_tr), y=y_tr).astype(float)

    pos, neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    sampler = WeightedRandomSampler([1/pos if y else 1/neg for y in y_tr],
                                    len(y_tr), replacement=True)

    tr_ld = DataLoader(tr_ds, batch_size=cfg.data.batch, sampler=sampler, num_workers=0)
    va_ld = DataLoader(va_ds, batch_size=cfg.data.batch, shuffle=False, num_workers=0)
    return tr_ld, va_ld, cls_w


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',       default='../conf/liteegtransformer.yaml')
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)

    x_paths = sorted(glob.glob(os.path.join(cfg.data.shift_dir, '*_X_shift.npy')))
    assert x_paths, f'No *_X_shift.npy under {cfg.data.shift_dir}'

    for x_path in x_paths:
        pid   = os.path.basename(x_path).split('_')[0]          # chb01
        if(pid!='chb01'):
            print(f'{pid} not chb01, continue')
            continue

        y_path = x_path.replace('_X_shift.npy', '_y_shift.npy')
        if not os.path.exists(y_path):
            print(f'[Skip] {pid}: missing y file'); continue

        # ---------- data ----------
        tr_ld, va_ld, cls_w = make_loaders(x_path, y_path, cfg)

        # ---------- model ----------
        model = LitEEGTransformer(
            lr=float(cfg.train.lr),
            in_ch=cfg.model.in_ch,
            out_ch=cfg.model.out_ch,
            T=cfg.model.T,
            class_weights=cls_w
        )

        # ---------- trainer ----------
        logger = pl_loggers.TensorBoardLogger('logs_liteegtransformer', name=pid)
        csv_logger = pl_loggers.CSVLogger('./logs_liteegtransformer/',  # 与上面同级
            name="LitEEGTransformer_CHBMIT_csv",  # 文件夹
            version=pid)
        trainer = pl.Trainer(
            max_epochs=cfg.train.epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=[cfg.train.gpu] if torch.cuda.is_available() else 1,
            logger=[logger,csv_logger],
            log_every_n_steps=10,
            enable_checkpointing=False
        )

        print(f'\n=== {pid}: {len(tr_ld.dataset)} train / {len(va_ld.dataset)} val ===')
        trainer.fit(model, tr_ld, va_ld)
        trainer.test(model, va_ld)

