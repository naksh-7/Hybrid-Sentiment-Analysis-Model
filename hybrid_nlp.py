import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

# Word2Vec via gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Utilities
# ----------------------------

from gensim.downloader import load as gensim_load

def load_glove(dim=100):
    # downloads pretrained glove.6B.100d
    glove = gensim_load(f'glove-wiki-gigaword-{dim}')
    return glove


def ensure_dirs():
    for d in ["checkpoints", "logs", "results"]:
        os.makedirs(d, exist_ok=True)


def in_kaggle() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def configure_caches_for_kaggle():
    if in_kaggle():
        os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/working/.cache/hf")
        os.environ.setdefault("HF_HOME", "/kaggle/working/.cache/hf")
        os.environ.setdefault("HF_DATASETS_CACHE", "/kaggle/working/.cache/hf_datasets")
        os.environ.setdefault("TORCH_HOME", "/kaggle/working/.cache/torch")
        os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
        os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
        os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
        print("[KAGGLE] Caches set to /kaggle/working/.cache/* for faster re-runs")


def save_metrics(y_true, y_pred, prefix: str):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    df = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1'],
        'value': [acc, prec, rec, f1],
        'model': [prefix]*4
    })
    out_csv = os.path.join('results', f'{prefix}_metrics.csv')
    df.to_csv(out_csv, index=False)

    rep = classification_report(y_true, y_pred, digits=4)
    out_rep = os.path.join('results', f'{prefix}_classification_report.txt')
    with open(out_rep, 'w', encoding='utf-8') as f:
        f.write(rep)

    # ✅ fixed print
    print(f"Saved metrics → {out_csv}\\nSaved report → {out_rep}")



# ----------------------------
# Data
# ----------------------------

@dataclass
class InputExample:
    text: str
    label: int


class TextDataset(Dataset):
    def __init__(self, examples: List[InputExample], bert_tok, max_len: int, word2idx: Dict[str, int], pad_idx: int):
        self.examples = examples
        self.bert_tok = bert_tok
        self.max_len = max_len
        self.word2idx = word2idx
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Transformer tokenization
        enc = self.bert_tok(
            ex.text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        # LSTM path: basic preprocess → indices
        tokens = simple_preprocess(ex.text, deacc=False, min_len=1)
        ids = [self.word2idx.get(t, self.word2idx.get('<unk>')) for t in tokens][:self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [self.pad_idx] * (self.max_len - len(ids))
        lstm_ids = torch.tensor(ids, dtype=torch.long)

        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'lstm_ids': lstm_ids,
            'label': torch.tensor(ex.label, dtype=torch.long)
        }
        return item


def load_splits(dataset: str) -> Tuple[List[InputExample], List[InputExample], str]:
    """Returns (train_examples, test_examples, dataset_tag). For SST-2, uses validation as test."""
    if dataset == 'imdb':
        ds = load_dataset('imdb')
        train, test = ds['train'], ds['test']
        def to_examples(split):
            out = []
            for r in split:
                text = r['text']
                label = int(r['label'])
                out.append(InputExample(text=text, label=label))
            return out
        return to_examples(train), to_examples(test), 'imdb'
    elif dataset == 'sst2':
        ds = load_dataset('glue', 'sst2')
        train, val = ds['train'], ds['validation']  # GLUE test is unlabeled
        def to_examples(split):
            out = []
            for r in split:
                text = r['sentence']
                label = int(r['label'])
                out.append(InputExample(text=text, label=label))
            return out
        return to_examples(train), to_examples(val), 'sst2'
    else:
        raise ValueError("Unsupported dataset. Choose from: imdb, sst2")


# ----------------------------
# Embeddings: Word2Vec + GloVe fusion
# ----------------------------

def build_vocab(texts: List[str], max_size: int = 40000, min_freq: int = 2) -> Tuple[Dict[str, int], List[str]]:
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(simple_preprocess(t, deacc=False, min_len=1))
    vocab = ['<pad>', '<unk>'] + [w for w, f in cnt.most_common() if f >= min_freq][:max_size]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return word2idx, vocab


def train_word2vec(texts: List[str], size: int = 100, window: int = 5, min_count: int = 2, workers: int = 4, epochs: int = 5) -> Word2Vec:
    toks = [simple_preprocess(t, deacc=False, min_len=1) for t in texts]
    model = Word2Vec(sentences=toks, vector_size=size, window=window, min_count=min_count, workers=workers, sg=1)
    model.train(toks, total_examples=len(toks), epochs=epochs)
    return model


def build_fused_embedding(word2idx: Dict[str, int], w2v: Word2Vec, glove_dim: int = 100, alpha: float = 0.6):
    """Return (emb_matrix, vocab_size, emb_dim).
    Policy:
    - if word in both: α*GloVe + (1-α)*Word2Vec
    - if only in GloVe: GloVe
    - if only in Word2Vec: Word2Vec
    - else: random normal(0, 0.1)
    """
    # ✅ load GloVe from gensim instead of torchtext
    glove = gensim_load(f'glove-wiki-gigaword-{glove_dim}')
    emb_dim = glove_dim
    vocab_size = len(word2idx)
    emb = np.random.normal(0, 0.1, size=(vocab_size, emb_dim)).astype(np.float32)

    for w, i in word2idx.items():
        if w == '<pad>':
            emb[i] = np.zeros(emb_dim, dtype=np.float32)
            continue

        g = glove[w] if w in glove else None
        v = w2v.wv[w] if w in w2v.wv else None

        if g is not None and v is not None:
            emb[i] = alpha * g + (1 - alpha) * v
        elif g is not None:
            emb[i] = g
        elif v is not None:
            emb[i] = v
        # else: keep random
    return emb, vocab_size, emb_dim


# ----------------------------
# Models
# ----------------------------

class BiLSTMEncoder(nn.Module):
    def __init__(self, emb_matrix: np.ndarray, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        vocab_size, emb_dim = emb_matrix.shape[0], emb_matrix.shape[1]
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.embedding.weight.requires_grad = False  # freeze; unfreeze later if desired
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.embedding(ids)
        out, _ = self.lstm(x)
        pooled, _ = torch.max(out, dim=1)
        return pooled  # [B, 2*hidden]


class DistilBERTEncoder(nn.Module):
    def __init__(self, model_name: str = 'distilbert-base-uncased', dropout: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / counts
        return self.drop(pooled)


class HybridClassifier(nn.Module):
    def __init__(self, emb_matrix: np.ndarray, bert_name: str = 'distilbert-base-uncased', lstm_hidden: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm_enc = BiLSTMEncoder(emb_matrix=emb_matrix, hidden_size=lstm_hidden)
        self.bert_enc = DistilBERTEncoder(bert_name)
        bert_hidden = self.bert_enc.bert.config.hidden_size
        fusion_dim = bert_hidden + (2 * lstm_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, batch):
        bert_vec = self.bert_enc(batch['input_ids'], batch['attention_mask'])
        lstm_vec = self.lstm_enc(batch['lstm_ids'])
        fused = torch.cat([bert_vec, lstm_vec], dim=1)
        logits = self.classifier(fused)
        return logits


class BERTBaseline(nn.Module):
    def __init__(self, bert_name: str = 'distilbert-base-uncased', num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.enc = DistilBERTEncoder(bert_name, dropout)
        hidden = self.enc.bert.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, batch):
        vec = self.enc(batch['input_ids'], batch['attention_mask'])
        return self.head(vec)


# ----------------------------
# Training & Evaluation
# ----------------------------

def train_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc='train', leave=False):
        for k in batch:
            batch[k] = batch[k].to(dev)
        labels = batch['label']
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.detach().cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return total_loss / len(loader.dataset), acc, f1


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval', leave=False):
            for k in batch:
                batch[k] = batch[k].to(dev)
            labels = batch['label']
            logits = model(batch)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.detach().cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return total_loss / len(loader.dataset), acc, f1, y_true, y_pred


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['imdb', 'sst2'], default='sst2')
    parser.add_argument('--model', choices=['hybrid', 'baseline_bert'], default='hybrid')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.6, help='blend factor for GloVe vs Word2Vec')
    parser.add_argument('--w2v_dim', type=int, default=100)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    ensure_dirs()
    configure_caches_for_kaggle()

    print(f"Loading dataset: {args.dataset}")
    train_ex, test_ex, tag_ds = load_splits(args.dataset)

    # Build vocab from train for LSTM path
    print("Building vocabulary…")
    word2idx, vocab = build_vocab([ex.text for ex in train_ex])
    pad_idx = word2idx['<pad>']

    # Train lightweight Word2Vec on train texts
    print("Training Word2Vec on training corpus…")
    w2v = train_word2vec([ex.text for ex in train_ex], size=args.w2v_dim)

    # Build fused embedding with GloVe
    print("Building fused GloVe+Word2Vec embedding matrix…")
    emb_matrix, vocab_size, emb_dim = build_fused_embedding(word2idx, w2v, glove_dim=args.w2v_dim, alpha=args.alpha)

    # Tokenizer for Transformer
    bert_name = 'distilbert-base-uncased'
    bert_tok = AutoTokenizer.from_pretrained(bert_name)

    # Datasets & Loaders
    train_ds = TextDataset(train_ex, bert_tok, args.max_len, word2idx, pad_idx)
    test_ds = TextDataset(test_ex, bert_tok, args.max_len, word2idx, pad_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    if args.model == 'hybrid':
        model = HybridClassifier(emb_matrix=emb_matrix, bert_name=bert_name, lstm_hidden=args.lstm_hidden)
        tag_model = 'hybrid'
    else:
        model = BERTBaseline(bert_name)
        tag_model = 'baseline_bert'
    model = model.to(dev)

    # Optim & Sched
    total_steps = len(train_loader) * args.epochs
    no_decay = ["bias", "LayerNorm.weight"]
    bert_params = []
    other_params = []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            bert_params.append({"params": [p], "weight_decay": 0.0})
        else:
            if 'bert' in n:
                bert_params.append({"params": [p], "weight_decay": 0.01})
            else:
                other_params.append(p)

    optimizer = optim.AdamW([
        *bert_params,
        {"params": other_params, "weight_decay": 0.01}
    ], lr=args.lr)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio * total_steps), num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_path = os.path.join('checkpoints', f'{tag_model}_{tag_ds}_best.pt')

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} [{tag_model} | {tag_ds}] on {dev}")
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        print(f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f}")
        te_loss, te_acc, te_f1, y_true, y_pred = eval_epoch(model, test_loader, criterion)
        print(f"Test : loss={te_loss:.4f} acc={te_acc:.4f} f1={te_f1:.4f}")
        if te_f1 > best_f1:
            best_f1 = te_f1
            torch.save({'state_dict': model.state_dict(), 'args': vars(args)}, best_path)
            print(f"Saved best checkpoint → {best_path}")

    # Final eval + save metrics
    te_loss, te_acc, te_f1, y_true, y_pred = eval_epoch(model, test_loader, criterion)
    prefix = f"{tag_model}_{tag_ds}"
    save_metrics(y_true, y_pred, prefix)

if __name__ == '__main__':
    import sys
    # Only reset args inside Jupyter notebooks
    if 'ipykernel_launcher' in sys.argv[0]:
        sys.argv = [sys.argv[0]]
    main()
