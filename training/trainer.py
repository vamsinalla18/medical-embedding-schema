# training/trainer.py
# Multi-task training loop — with LR scheduler + early stopping

import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from training.losses import MultiTaskLoss
from utils.negative_sampling import sample_negative_batch


# ── Token IDs ─────────────────────────────────────────────────────────────────
PAD_TOKEN_ID  = 0
UNK_TOKEN_ID  = 1
CLS_TOKEN_ID  = 2
SEP_TOKEN_ID  = 3
MASK_TOKEN_ID = 4


def mask_tokens(input_ids: torch.Tensor, vocab_size: int, mask_prob: float = 0.15):
    labels      = input_ids.clone()
    prob_matrix = torch.full(input_ids.shape, mask_prob)
    masked      = torch.bernoulli(prob_matrix).bool()
    masked     &= input_ids >= 5
    labels[~masked]   = -100
    input_ids[masked] = MASK_TOKEN_ID
    return input_ids, labels


def encode_entity(entity_name: str, tokenizer, max_len: int = 16):
    enc  = tokenizer.encode(entity_name.replace("_", " "))
    ids  = [CLS_TOKEN_ID] + enc.ids[: max_len - 2] + [SEP_TOKEN_ID]
    ids += [PAD_TOKEN_ID] * (max_len - len(ids))
    return torch.tensor(ids[:max_len], dtype=torch.long)


def train(
    model,
    tokenizer,
    corpus,
    triples,
    entity_types,
    entities,
    relations,
    entity_to_id,
    relation_to_id,
    num_epochs     = 30,
    lr             = 1e-4,
    device         = "cpu",
    batch_size     = 8,
    rel_embed_path = "relation_embeds.pt",
    patience       = 5,        # early stopping patience
):
    """
    Multi-task training with:
      - LR scheduler (ReduceLROnPlateau) — halves LR when loss stalls
      - Early stopping — stops if no improvement for `patience` epochs
      - Saves trained relation embeddings

    Returns:
        model, relation_embeds
    """
    from data.kg_triples import TYPE_TO_ID

    model     = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MultiTaskLoss(lambda1=0.5, lambda2=0.3)

    # LR scheduler — reduce LR by 0.5 if val loss doesn't improve for 2 epochs
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # Relation embeddings
    relation_embeds = nn.Embedding(len(relations), model.embed_dim).to(device)
    optimizer.add_param_group({"params": relation_embeds.parameters()})

    vocab_size = tokenizer.get_vocab_size()

    # Early stopping state
    best_loss      = float("inf")
    best_model_state    = None
    best_rel_state = None
    no_improve     = 0

    print(f"  Training config: {num_epochs} epochs | "
          f"LR={lr} | batch={batch_size} | patience={patience}")
    print(f"  Corpus: {len(corpus)} sentences | "
          f"Triples: {len(triples)} | Entities: {len(entities)}")

    for epoch in range(num_epochs):
        model.train()
        relation_embeds.train()
        total_loss = 0
        steps      = 0

        random.shuffle(corpus)
        for i in tqdm(range(0, len(corpus) - batch_size, batch_size),
                      desc=f"Epoch {epoch+1}/{num_epochs}"):

            batch_texts = corpus[i: i + batch_size]
            encodings   = [tokenizer.encode(t) for t in batch_texts]

            max_len   = min(max(len(e.ids) + 2 for e in encodings), 128)
            input_ids = []
            for enc in encodings:
                ids  = [CLS_TOKEN_ID] + enc.ids[: max_len - 2] + [SEP_TOKEN_ID]
                ids += [PAD_TOKEN_ID] * (max_len - len(ids))
                input_ids.append(ids[:max_len])

            input_ids  = torch.tensor(input_ids, dtype=torch.long).to(device)
            masked_ids, mlm_labels = mask_tokens(input_ids.clone(), vocab_size)

            mlm_logits, _, _ = model(masked_ids)

            # KG batch
            batch_triples = random.sample(triples, min(batch_size, len(triples)))
            neg_tails     = sample_negative_batch(batch_triples, all_entities=entities)

            def get_entity_embed(name):
                ids = encode_entity(name, tokenizer).unsqueeze(0).to(device)
                _, emb, _ = model(ids)
                return emb

            h_embeds     = torch.cat([get_entity_embed(h) for h, r, t in batch_triples], dim=0)
            t_embeds     = torch.cat([get_entity_embed(t) for h, r, t in batch_triples], dim=0)
            t_neg_embeds = torch.cat([get_entity_embed(neg) for neg in neg_tails], dim=0)
            r_ids        = torch.tensor(
                [relation_to_id[r] for _, r, _ in batch_triples], dtype=torch.long
            ).to(device)
            r_embeds_batch = relation_embeds(r_ids)

            type_labels = torch.tensor(
                [TYPE_TO_ID[entity_types[h]] for h, r, t in batch_triples],
                dtype=torch.long
            ).to(device)
            _, _, type_logits_kg = model(
                torch.stack([encode_entity(h, tokenizer).to(device)
                             for h, r, t in batch_triples])
            )

            loss, breakdown = criterion(
                mlm_logits, mlm_labels,
                h_embeds, r_embeds_batch, t_embeds, t_neg_embeds,
                type_logits_kg, type_labels
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps      += 1

        avg_loss = total_loss / max(steps, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"{breakdown}")

        # LR scheduler step
        scheduler.step(avg_loss)

        # Early stopping check
        if avg_loss < best_loss - 1e-4:
            best_loss       = avg_loss
            best_model_state     = {k: v.clone() for k, v in model.state_dict().items()}
            best_rel_state  = {k: v.clone() for k, v in relation_embeds.state_dict().items()}
            no_improve      = 0
            print(f"  ✓ Best loss updated: {best_loss:.4f}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    if best_model_state:
        model.load_state_dict(best_model_state)
        relation_embeds.load_state_dict(best_rel_state)
        print(f"  Restored best model (loss={best_loss:.4f})")

    # Save relation embeddings
    torch.save(relation_embeds.state_dict(), rel_embed_path)
    print(f"  Relation embeddings saved to: {rel_embed_path}")
    print("Training complete.")

    return model, relation_embeds
