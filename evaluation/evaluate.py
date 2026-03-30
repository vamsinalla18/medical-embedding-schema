# evaluation/evaluate.py
# Evaluation: Link Prediction (MRR, Hits@10) + t-SNE Clustering

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from data.kg_triples import TYPE_TO_ID

# ── Token IDs ─────────────────────────────────────────────────────────────────
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3

ENTITY_MAX_LEN = 16


def _encode_entity_ids(entity_name: str, tokenizer, max_len: int = ENTITY_MAX_LEN) -> list:
    """Tokenize entity → [CLS] + ids + [SEP] + padding."""
    enc = tokenizer.encode(entity_name.replace("_", " "))
    ids = [CLS_TOKEN_ID] + enc.ids[: max_len - 2] + [SEP_TOKEN_ID]
    ids += [PAD_TOKEN_ID] * (max_len - len(ids))
    return ids[:max_len]


def get_all_entity_embeddings(model, tokenizer, entities, device="cpu"):
    """Compute CLS embeddings for every entity."""
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for entity in entities:
            ids       = _encode_entity_ids(entity, tokenizer)
            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            _, cls_embed, _ = model(input_ids)
            embeddings[entity] = cls_embed.squeeze(0).cpu().numpy()
    return embeddings


def link_prediction(model, tokenizer, relation_embeds,
                    triples, entities, relation_to_id, device="cpu"):
    """
    Evaluate link prediction: given (h, r, ?), rank correct tail.
    Returns MRR and Hits@10.
    """
    model.eval()
    entity_embeds = get_all_entity_embeddings(model, tokenizer, entities, device)

    mrr_scores = []
    hits_at_10 = []

    with torch.no_grad():
        for head, rel, tail in triples:
            if rel not in relation_to_id:
                continue

            h_emb = torch.tensor(entity_embeds[head]).to(device)
            r_id  = torch.tensor(relation_to_id[rel]).to(device)
            r_emb = relation_embeds(r_id.unsqueeze(0)).squeeze(0)
            query = h_emb + r_emb

            scores = []
            for candidate in entities:
                c_emb = torch.tensor(entity_embeds[candidate]).to(device)
                score = -torch.norm(query - c_emb).item()
                scores.append((score, candidate))

            scores.sort(reverse=True)
            ranked = [name for _, name in scores]

            rank = ranked.index(tail) + 1 if tail in ranked else len(entities) + 1
            mrr_scores.append(1.0 / rank)
            hits_at_10.append(1 if rank <= 10 else 0)

    mrr = np.mean(mrr_scores)
    h10 = np.mean(hits_at_10)
    print(f"Link Prediction | MRR: {mrr:.4f} | Hits@10: {h10:.4f}")
    return mrr, h10


def type_classification_accuracy(model, tokenizer,
                                  entities, entity_types, device="cpu"):
    """Evaluate entity type classification accuracy."""
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for entity in entities:
            ids       = _encode_entity_ids(entity, tokenizer)
            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            _, _, type_logits = model(input_ids)

            pred = type_logits.argmax(dim=-1).item()
            true = TYPE_TO_ID[entity_types[entity]]
            if pred == true:
                correct += 1
            total += 1

    acc = correct / total
    print(f"Type Classification Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def plot_tsne(model, tokenizer, entities, entity_types,
              device="cpu", save_path="tsne_plot.png"):
    """Visualize entity embeddings with t-SNE, coloured by entity type."""
    entity_embeds = get_all_entity_embeddings(model, tokenizer, entities, device)

    names   = list(entity_embeds.keys())
    vectors = np.array([entity_embeds[n] for n in names])
    types   = [entity_types[n] for n in names]

    # PCA to speed up t-SNE when embed_dim is large
    if vectors.shape[1] > 50:
        n_components = min(50, vectors.shape[0] - 1, vectors.shape[1])
        vectors = PCA(n_components=n_components).fit_transform(vectors)

    perplexity = min(5, len(names) - 1)
    tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(vectors)

    type_colors = {
        "disease":  "red",
        "drug":     "blue",
        "symptom":  "green",
        "anatomy":  "orange",
    }
    colors = [type_colors.get(t, "gray") for t in types]

    plt.figure(figsize=(12, 9))
    for i, (x, y) in enumerate(coords):
        plt.scatter(x, y, c=colors[i], s=80)
        plt.annotate(names[i], (x, y), fontsize=7, alpha=0.8)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=t) for t, c in type_colors.items()]
    plt.legend(handles=legend_handles)
    plt.title("t-SNE: Medical Entity Embeddings")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"t-SNE plot saved to: {save_path}")
    plt.show()
