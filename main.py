# main.py
# Entry point: full Medical Embedding pipeline

import torch
import torch.nn as nn
import os

from data.corpus import load_corpus, preprocess
from data.kg_triples import build_kg, TYPE_TO_ID
from models.tokenizer import train_tokenizer, load_tokenizer
from models.transformer import MedicalTransformerEncoder
from training.trainer import train
from evaluation.evaluate import link_prediction, type_classification_accuracy, plot_tsne


# ── Config ────────────────────────────────────────────────────────────────────
EMBED_DIM      = 256
NUM_LAYERS     = 6
NUM_HEADS      = 8
FF_DIM         = 1024
MAX_LEN        = 128
VOCAB_SIZE     = 15000
NUM_EPOCHS     = 50        # trainer will early-stop before this if loss plateaus
LR             = 1e-4
BATCH_SIZE     = 8
PATIENCE       = 5         # early stopping patience
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER_PATH = "tokenizer_output"
MODEL_PATH     = "medical_encoder.pt"
REL_EMBED_PATH = "relation_embeds.pt"
USE_PUBMED     = True
PUBMED_SIZE    = 50000     # load up to 50k samples
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"Device: {DEVICE}")

    # 1. Load corpus
    print("\n[1/6] Loading corpus...")
    raw    = load_corpus(use_pubmed=USE_PUBMED, pubmed_size=PUBMED_SIZE)
    corpus = preprocess(raw)
    print(f"  Corpus size: {len(corpus)} sentences")

    # 2. Build KG
    print("\n[2/6] Building Knowledge Graph...")
    triples, entity_types, entities, relations, entity_to_id, relation_to_id = build_kg(corpus)
    print(f"  Entities : {len(entities)}")
    print(f"  Relations: {len(relations)} — {relations}")
    print(f"  Triples  : {len(triples)}")

    # 3. Tokenizer
    print("\n[3/6] Tokenizer...")
    if os.path.exists(os.path.join(TOKENIZER_PATH, "tokenizer.json")):
        tokenizer = load_tokenizer(TOKENIZER_PATH)
    else:
        tokenizer = train_tokenizer(corpus, vocab_size=VOCAB_SIZE, save_path=TOKENIZER_PATH)
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")

    # 4. Model
    print("\n[4/6] Initializing model...")
    model = MedicalTransformerEncoder(
        vocab_size = tokenizer.get_vocab_size(),
        embed_dim  = EMBED_DIM,
        num_layers = NUM_LAYERS,
        num_heads  = NUM_HEADS,
        ff_dim     = FF_DIM,
        max_len    = MAX_LEN,
        num_types  = len(TYPE_TO_ID),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # 5. Train
    print("\n[5/6] Training...")
    model, relation_embeds = train(
        model          = model,
        tokenizer      = tokenizer,
        corpus         = corpus,
        triples        = triples,
        entity_types   = entity_types,
        entities       = entities,
        relations      = relations,
        entity_to_id   = entity_to_id,
        relation_to_id = relation_to_id,
        num_epochs     = NUM_EPOCHS,
        lr             = LR,
        device         = DEVICE,
        batch_size     = BATCH_SIZE,
        rel_embed_path = REL_EMBED_PATH,
        patience       = PATIENCE,
    )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    # 6. Evaluate
    print("\n[6/6] Evaluating...")
    relation_embeds = relation_embeds.to(DEVICE)

    mrr, h10 = link_prediction(
        model, tokenizer, relation_embeds,
        triples        = triples,
        entities       = entities,
        relation_to_id = relation_to_id,
        device         = DEVICE,
    )
    acc = type_classification_accuracy(
        model, tokenizer,
        entities     = entities,
        entity_types = entity_types,
        device       = DEVICE,
    )
    plot_tsne(
        model, tokenizer,
        entities     = entities,
        entity_types = entity_types,
        device       = DEVICE,
        save_path    = "tsne_plot.png",
    )

    print("\n── Final Results ──────────────────────────")
    print(f"  MRR:                    {mrr:.4f}")
    print(f"  Hits@10:                {h10:.4f}")
    print(f"  Type Classification:    {acc:.4f}")
    print(f"  Entities:               {len(entities)}")
    print(f"  Triples:                {len(triples)}")
    print(f"  Vocab size:             {tokenizer.get_vocab_size()}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()
