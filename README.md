# Hybrid-Sentiment-Analysis-Model
# Hybrid NLP Sentiment Classification (HCI Assignment)

This project explores hybrid embedding techniques for sentiment classification on **IMDB** and **SST-2 (GLUE)** datasets.  
It was developed as part of my **Human-Computer Interaction (HCI) assignment**.

## Features
- Train/test on **IMDB** or **SST-2**
- Word2Vec trained on corpus + pre-trained GloVe (6B-100d) → fused embeddings
- DistilBERT encoder for contextual features
- Hybrid model: **[BiLSTM pooled ⊕ DistilBERT pooled] → MLP classifier**
- Baseline: **DistilBERT-only classifier**
- Metrics: Accuracy, Precision, Recall, F1 (saved as CSV/JSON)

## 📊 Sample Results
| Dataset | Model          | Accuracy | Precision | Recall | F1   |
|---------|---------------|----------|-----------|--------|------|
| IMDB    | Hybrid        | 0.90     | 0.89      | 0.91   | 0.90 |
| IMDB    | Baseline BERT | 0.88     | 0.87      | 0.88   | 0.88 |
| SST-2   | Hybrid        | 0.89     | 0.88      | 0.90   | 0.89 |
| SST-2   | Baseline BERT | 0.87     | 0.86      | 0.87   | 0.87 |

## 🛠 Installation
```bash
pip install -r requirements.txt
