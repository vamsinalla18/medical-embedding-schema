# utils/negative_sampling.py
# Corrupt triples for relation prediction training

import random
from data.kg_triples import ENTITIES, ENTITY_TO_ID


def corrupt_triple(head: str, relation: str, tail: str, all_entities=ENTITIES, mode="tail"):
    """
    Generate a negative sample by replacing the head or tail with a random entity.

    Args:
        head: head entity name
        relation: relation string
        tail: tail entity name
        all_entities: list of all entity names
        mode: 'tail' or 'head' — which side to corrupt

    Returns:
        (head, relation, corrupted_entity) or (corrupted_entity, relation, tail)
    """
    while True:
        neg = random.choice(all_entities)
        if mode == "tail" and neg != tail:
            return head, relation, neg
        elif mode == "head" and neg != head:
            return neg, relation, tail


def sample_negative_batch(triples: list, all_entities=ENTITIES):
    """
    For each triple in the batch, sample one negative triple.

    Args:
        triples: list of (head, relation, tail) tuples
        all_entities: pool of entities to sample from

    Returns:
        List of negative tail entity names
    """
    neg_tails = []
    for head, rel, tail in triples:
        mode = random.choice(["head", "tail"])
        _, _, neg = corrupt_triple(head, rel, tail, all_entities, mode="tail")
        neg_tails.append(neg)
    return neg_tails
