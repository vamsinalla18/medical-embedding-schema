# training/trainer.py
# Multi-task training loop for the medical embedding model

import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from training.losses import MultiTaskLoss
from utils.negative_sampling import sample_negative_batch


# ── Token IDs — must match models/tokenizer.py SPECIAL_TOKENS order ──────────
PAD_TOKEN_ID  = 0   # [PAD]
UNK_TOKEN_ID  = 1   # [UNK]
CLS_TOKEN_ID  = 2   # [CLS]
SEP_TOKEN_ID  = 3   # [SEP]
MASK_TOKEN_ID = 4   # [MASK]


def mask_tokens(input_ids: torch.Tensor, vocab_size: int, mask_prob: float = 0.15):
    """
    Apply random masking for MLM.
    Returns masked input_ids and labels (-100 for non-masked positions).
    """
    labels       = input_ids.clone()
    prob_matrix  = torch.full(input_ids.shape, mask_prob)
    masked       = torch.bernoulli(prob_matrix).bool()

    # Don't mask special tokens (ids 0-4)
    masked &= input_ids >= 5

    labels[~masked]    = -100
    input_ids[masked]  = MASK_TOKEN_ID
    return input_ids, labels


def encode_entity(entity_name: str, tokenizer, max_len: int = 16):
    """
    Tokenize entity name → [CLS] + ids + [SEP] + padding.
    Returns a 1-D LongTensor of length max_len.
    """
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
    num_epochs  = 20,
    lr          = 1e-4,
    device      = "cpu",
    batch_size  = 8,
    rel_embed_path = "relation_embeds.pt",
):
    """
    Full multi-task training loop.

    Returns:
        model           : trained MedicalTransformerEncoder
        relation_embeds : trained nn.Embedding for relations
    """
    from data.kg_triples import TYPE_TO_ID

    model     = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = MultiTaskLoss(lambda1=0.5, lambda2=0.3)

    # ── Relation embeddings (learnable, trained alongside model) ───────────────
    relation_embeds = nn.Embedding(len(relations), model.embed_dim).to(device)
    optimizer.add_param_group({"params": relation_embeds.parameters()})

    vocab_size = tokenizer.get_vocab_size()

    for epoch in range(num_epochs):
        model.train()
        relation_embeds.train()
        total_loss = 0
        steps      = 0

        # ── MLM batches from corpus ───────────────────────────────────────────
        random.shuffle(corpus)
        for i in tqdm(range(0, len(corpus) - batch_size, batch_size), desc=f"Epoch {epoch+1}"):
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

            # Forward pass
            mlm_logits, cls_embed, type_logits = model(masked_ids)

            # ── KG batch ─────────────────────────────────────────────────────
            batch_triples = random.sample(triples, min(batch_size, len(triples)))
            neg_tails     = sample_negative_batch(batch_triples, all_entities=entities)

            def get_entity_embed(name):
                ids = encode_entity(name, tokenizer).unsqueeze(0).to(device)
                _, emb, _ = model(ids)
                return emb

            h_embeds     = torch.cat([get_entity_embed(h) for h, r, t in batch_triples], dim=0)
            t_embeds     = torch.cat([get_entity_embed(t) for h, r, t in batch_triples], dim=0)
            t_neg_embeds = torch.cat([get_entity_embed(neg) for neg in neg_tails],        dim=0)

            r_ids    = torch.tensor(
                [relation_to_id[r] for _, r, _ in batch_triples], dtype=torch.long
            ).to(device)
            r_embeds = relation_embeds(r_ids)   # ← uses the TRAINED relation embeddings

            # Type labels from head entity
            type_labels = torch.tensor(
                [TYPE_TO_ID[entity_types[h]] for h, r, t in batch_triples],
                dtype=torch.long
            ).to(device)
            _, _, type_logits_kg = model(
                torch.stack([encode_entity(h, tokenizer).to(device) for h, r, t in batch_triples])
            )

            # Combined loss
            loss, breakdown = criterion(
                mlm_logits, mlm_labels,
                h_embeds, r_embeds, t_embeds, t_neg_embeds,
                type_logits_kg, type_labels
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps      += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | {breakdown}")

    # ── Save trained relation embeddings ──────────────────────────────────────
    torch.save(relation_embeds.state_dict(), rel_embed_path)
    print(f"  Relation embeddings saved to: {rel_embed_path}")

    print("Training complete.")

    # Return BOTH model and trained relation embeddings
    return model, relation_embeds
