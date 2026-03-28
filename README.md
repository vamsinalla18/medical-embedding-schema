# Medical Embedding Schema using Transformer and Knowledge Graphs

A transformer-based embedding model trained from scratch for structured medical concept representation.

## Project Structure

```
medical_embedding/
├── data/
│   ├── kg_triples.py        # Knowledge graph triples & entity types
│   └── corpus.py            # Text corpus loading & preprocessing
├── models/
│   ├── tokenizer.py         # WordPiece/BPE tokenizer training
│   └── transformer.py       # Transformer encoder (built from scratch)
├── training/
│   ├── losses.py            # MLM, Relation, Type classification losses
│   └── trainer.py           # Multi-task training loop
├── evaluation/
│   └── evaluate.py          # MRR, Hits@10, clustering (t-SNE)
├── utils/
│   └── negative_sampling.py # Negative triple sampling
├── main.py                  # Entry point — runs full pipeline
└── requirements.txt         # Dependencies
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

## Pipeline

1. Load medical corpus + KG triples
2. Train tokenizer on corpus
3. Train transformer with multi-task loss (MLM + Relation + Type)
4. Evaluate via link prediction and clustering
