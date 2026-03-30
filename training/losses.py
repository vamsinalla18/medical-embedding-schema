# training/losses.py
# All three loss functions built from scratch using raw PyTorch tensor ops.
# NO nn.CrossEntropyLoss, NO F.cosine_similarity, NO F.softmax, NO F.log_softmax.
# Every formula is implemented explicitly.

import torch
import torch.nn as nn
import math


# ── Utility functions ──────────────────────────────────────────────────────────

def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable log-softmax from scratch.
    Formula: x - max(x) - log(sum(exp(x - max(x))))

    Using log-sum-exp trick to avoid overflow.
    Equivalent to F.log_softmax but manual.
    """
    x_max  = x.max(dim=dim, keepdim=True).values
    x_shift = x - x_max
    log_sum = torch.log(torch.exp(x_shift).sum(dim=dim, keepdim=True))
    return x_shift - log_sum


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity between two tensors along last dimension.
    Formula: dot(a, b) / (||a|| * ||b||)

    Built from scratch — no F.cosine_similarity.
    """
    dot      = (a * b).sum(dim=-1)                      # element-wise multiply then sum
    norm_a   = torch.sqrt((a * a).sum(dim=-1) + eps)    # L2 norm of a
    norm_b   = torch.sqrt((b * b).sum(dim=-1) + eps)    # L2 norm of b
    return dot / (norm_a * norm_b)


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Euclidean distance between two tensors along last dimension.
    Formula: sqrt(sum((a - b)^2))

    Built from scratch.
    """
    diff = a - b
    return torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)


# ── Loss 1: Masked Language Modeling Loss ─────────────────────────────────────

class MLMLoss(nn.Module):
    """
    Masked Language Modeling (MLM) loss — built from scratch.

    Standard cross-entropy loss:
        L_MLM = -sum( y * log_softmax(logits) )

    Only computed on masked positions (label != ignore_index).

    Manual steps:
        1. Compute log_softmax over vocab dimension.
        2. Gather the log-prob of the correct token.
        3. Average over all masked positions (ignore padding with ignore_index=-100).
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, T, vocab_size) — raw scores for each token position
            labels : (B, T)             — correct token ids; -100 = not masked (ignore)

        Returns:
            scalar loss
        """
        B, T, V = logits.shape

        # Flatten to (B*T, V) and (B*T,) for easier indexing
        logits_flat = logits.view(B * T, V)
        labels_flat = labels.view(B * T)

        # Compute log-softmax over vocab dimension (manual, numerically stable)
        log_probs = log_softmax(logits_flat, dim=-1)   # (B*T, V)

        # Create mask for positions we actually want to compute loss on
        mask = (labels_flat != self.ignore_index)       # (B*T,) boolean

        # Extract log-prob of the correct label at each masked position
        # labels_flat[mask]: indices into vocab for valid positions
        valid_labels    = labels_flat[mask].clamp(min=0)   # guard against -100
        valid_log_probs = log_probs[mask]                  # (num_masked, V)

        # Gather the log-prob of the true token:  log_probs[i, label[i]]
        # Shape: (num_masked,)
        correct_log_probs = valid_log_probs[
            torch.arange(valid_labels.size(0), device=logits.device),
            valid_labels
        ]

        # NLL loss = -mean(log P(correct token))
        loss = -correct_log_probs.mean()
        return loss


# ── Loss 2: Relation Prediction Loss (TransE-style margin loss) ───────────────

class RelationPredictionLoss(nn.Module):
    """
    Margin-based relation prediction loss — built from scratch.

    Enforces the TransE constraint: h + r ≈ t

    Formula:
        L_rel = mean( max(0, gamma + d(h+r, t) - d(h+r, t')) )

    where:
        h = head entity embedding
        r = relation embedding
        t = correct tail embedding (positive sample)
        t' = corrupted tail embedding (negative sample)
        d = distance function (cosine or euclidean)
        gamma = margin hyperparameter

    Intuition:
        We want d(h+r, t) to be SMALL (close to correct tail)
        and d(h+r, t') to be LARGE (far from wrong tail).
        The margin gamma ensures a minimum gap between them.
        When d_pos < d_neg - gamma, loss = 0 (already well separated).
    """

    def __init__(self, gamma: float = 1.0, distance: str = "cosine"):
        super().__init__()
        self.gamma    = gamma
        self.distance = distance

    def compute_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine distance = 1 - cosine_similarity (both built from scratch)."""
        if self.distance == "cosine":
            return 1.0 - cosine_similarity(a, b)
        else:
            return euclidean_distance(a, b)

    def forward(
        self,
        h_embed:     torch.Tensor,
        r_embed:     torch.Tensor,
        t_embed:     torch.Tensor,
        t_neg_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_embed     : (B, D) — head entity embeddings
            r_embed     : (B, D) — relation embeddings
            t_embed     : (B, D) — positive tail embeddings
            t_neg_embed : (B, D) — negative (corrupted) tail embeddings

        Returns:
            scalar loss
        """
        query = h_embed + r_embed                            # TransE: h + r

        d_pos = self.compute_distance(query, t_embed)        # (B,) dist to correct tail
        d_neg = self.compute_distance(query, t_neg_embed)    # (B,) dist to wrong tail

        # Margin loss: penalize when positive is NOT closer than negative by margin gamma
        # max(0, ...) is relu — implemented manually
        margin_loss = d_pos - d_neg + self.gamma
        loss = torch.where(margin_loss > 0, margin_loss, torch.zeros_like(margin_loss))

        return loss.mean()


# ── Loss 3: Type Classification Loss ──────────────────────────────────────────

class TypeClassificationLoss(nn.Module):
    """
    Entity type classification loss — built from scratch.

    Standard cross-entropy for multi-class classification:
        L_type = -mean( log P(correct_class) )
               = -mean( log_softmax(logits)[correct_class] )

    Manual steps:
        1. log_softmax over class dimension.
        2. Index into correct class per sample.
        3. Negate and average.
    """

    def __init__(self):
        super().__init__()

    def forward(self, type_logits: torch.Tensor, type_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            type_logits : (B, num_types) — raw scores per entity type
            type_labels : (B,)           — correct class index (0-3)

        Returns:
            scalar loss
        """
        # log-softmax over class dimension (manual)
        log_probs = log_softmax(type_logits, dim=-1)   # (B, num_types)

        # Gather log-prob of correct class for each sample in batch
        B = type_labels.size(0)
        correct_log_probs = log_probs[
            torch.arange(B, device=type_logits.device),
            type_labels
        ]                                               # (B,)

        # NLL = -mean(log P(correct class))
        return -correct_log_probs.mean()


# ── Combined Multi-Task Loss ───────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss:
        L = L_mlm + lambda1 * L_relation + lambda2 * L_type

    All three sub-losses are built from scratch (no nn.* loss functions).

    lambda1 and lambda2 are scalar weights that balance the three objectives.
    """

    def __init__(self, lambda1: float = 0.5, lambda2: float = 0.3):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.mlm_loss      = MLMLoss()
        self.relation_loss = RelationPredictionLoss(gamma=1.0, distance="cosine")
        self.type_loss     = TypeClassificationLoss()

    def forward(
        self,
        mlm_logits:   torch.Tensor,
        mlm_labels:   torch.Tensor,
        h_embed:      torch.Tensor,
        r_embed:      torch.Tensor,
        t_embed:      torch.Tensor,
        t_neg_embed:  torch.Tensor,
        type_logits:  torch.Tensor,
        type_labels:  torch.Tensor,
    ):
        """
        Returns:
            total  : scalar combined loss
            breakdown : dict with individual loss values for logging
        """
        l_mlm  = self.mlm_loss(mlm_logits, mlm_labels)
        l_rel  = self.relation_loss(h_embed, r_embed, t_embed, t_neg_embed)
        l_type = self.type_loss(type_logits, type_labels)

        total = l_mlm + self.lambda1 * l_rel + self.lambda2 * l_type

        breakdown = {
            "mlm":      round(l_mlm.item(),  4),
            "relation": round(l_rel.item(),  4),
            "type":     round(l_type.item(), 4),
        }
        return total, breakdown


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, V, D, C = 4, 16, 500, 64, 4

    # MLM loss test
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, ::3] = -100        # every 3rd position is not masked
    mlm = MLMLoss()
    print("MLM loss       :", mlm(logits, labels).item())

    # Relation loss test
    h = torch.randn(B, D)
    r = torch.randn(B, D)
    t = torch.randn(B, D)
    t_neg = torch.randn(B, D)
    rel = RelationPredictionLoss()
    print("Relation loss  :", rel(h, r, t, t_neg).item())

    # Type loss test
    type_logits = torch.randn(B, C)
    type_labels = torch.randint(0, C, (B,))
    typ = TypeClassificationLoss()
    print("Type loss      :", typ(type_logits, type_labels).item())

    # Combined
    multi = MultiTaskLoss()
    total, breakdown = multi(logits, labels, h, r, t, t_neg, type_logits, type_labels)
    print("Total loss     :", total.item())
    print("Breakdown      :", breakdown)