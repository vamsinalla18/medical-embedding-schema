# data/corpus.py
# Medical text corpus loading and preprocessing
# Supports real PubMed abstracts from HuggingFace + synthetic fallback

import re


# ── Synthetic fallback corpus (used if internet/datasets unavailable) ─────────
SYNTHETIC_CORPUS = [
    "Diabetes is a chronic metabolic disease characterized by elevated blood sugar levels.",
    "Type 2 diabetes is commonly treated with metformin and insulin therapy.",
    "Insulin is a hormone produced by the pancreas that regulates blood sugar.",
    "Hyperglycemia, or high blood sugar, is the primary symptom of uncontrolled diabetes.",
    "Fatigue and polyuria are common early symptoms of diabetes mellitus.",
    "Asthma is a chronic respiratory disease that causes inflammation of the airways.",
    "Patients with asthma often use inhalers to manage wheezing and breathing difficulties.",
    "Asthma primarily affects the lungs and bronchial passages.",
    "Hypertension, or high blood pressure, is a major risk factor for cardiovascular disease.",
    "Beta blockers are commonly prescribed to treat hypertension and heart conditions.",
    "Headache and dizziness are frequent symptoms of uncontrolled hypertension.",
    "Pneumonia is a serious lung infection that causes fever, cough, and chest pain.",
    "Bacterial pneumonia is typically treated with a course of antibiotics.",
    "Fever is a common symptom associated with respiratory infections like pneumonia.",
    "The cardiovascular system includes the heart and blood vessels.",
    "The respiratory system includes the lungs, bronchi, and trachea.",
    "Blood sugar regulation is critical for metabolic homeostasis.",
    "Ibuprofen is a nonsteroidal anti-inflammatory drug used to reduce fever and pain.",
    "Chills and sweating often accompany high fever in infectious diseases.",
    "Medical knowledge graphs capture structured relationships between diseases, drugs, and symptoms.",
]


# ── PubMed loader ─────────────────────────────────────────────────────────────

def load_pubmed(num_samples: int = 5000) -> list:
    """
    Load real PubMed abstracts from HuggingFace datasets.
    Uses the 'pubmed_qa' dataset which is free and well-structured.

    Args:
        num_samples: how many abstracts to load (default 5000)

    Returns:
        list of abstract strings
    """
    try:
        from datasets import load_dataset

        print(f"  Downloading PubMed abstracts ({num_samples} samples)...")

        # pubmed_qa is reliable and contains real medical abstracts
        dataset = load_dataset(
            "pubmed_qa",
            "pqa_unlabeled",
            split="train"
        )

        texts = []
        for item in dataset:
            # each item has a 'context' field with a list of sentences
            if "context" in item and item["context"]:
                context = item["context"]
                # context is a dict with 'contexts' key (list of paragraph strings)
                if isinstance(context, dict) and "contexts" in context:
                    for para in context["contexts"]:
                        if isinstance(para, str) and len(para.strip()) > 30:
                            texts.append(para.strip())
                elif isinstance(context, list):
                    for para in context:
                        if isinstance(para, str) and len(para.strip()) > 30:
                            texts.append(para.strip())

            if len(texts) >= num_samples:
                break

        if len(texts) < 100:
            raise ValueError(f"Too few samples extracted: {len(texts)}")

        print(f"  Loaded {len(texts)} PubMed paragraphs.")
        return texts[:num_samples]

    except Exception as e:
        print(f"  PubMed load failed: {e}")
        print("  Trying alternative dataset...")
        return load_pubmed_alternative(num_samples)


def load_pubmed_alternative(num_samples: int = 5000) -> list:
    """
    Fallback: load from 'medical_questions_pairs' or 'mtsamples' on HuggingFace.
    """
    try:
        from datasets import load_dataset

        print("  Trying 'medalpaca/medical_meadow_pubmed_causal'...")
        dataset = load_dataset(
            "medalpaca/medical_meadow_pubmed_causal",
            split="train"
        )

        texts = []
        for item in dataset:
            for field in ["input", "output", "instruction"]:
                val = item.get(field, "")
                if isinstance(val, str) and len(val.strip()) > 30:
                    texts.append(val.strip())
            if len(texts) >= num_samples:
                break

        if len(texts) < 100:
            raise ValueError(f"Too few samples: {len(texts)}")

        print(f"  Loaded {len(texts)} samples from medical_meadow_pubmed_causal.")
        return texts[:num_samples]

    except Exception as e:
        print(f"  Alternative load also failed: {e}")
        print("  Falling back to synthetic corpus.")
        return SYNTHETIC_CORPUS


# ── Main loader ───────────────────────────────────────────────────────────────

def load_corpus(use_pubmed: bool = True, pubmed_size: int = 5000) -> list:
    """
    Load the medical text corpus.

    Args:
        use_pubmed : if True, load real PubMed abstracts (requires internet)
        pubmed_size: number of PubMed samples to load

    Returns:
        list of raw text strings
    """
    if use_pubmed:
        texts = load_pubmed(pubmed_size)
        # If we got real data, return it
        if texts and texts is not SYNTHETIC_CORPUS:
            return texts
        # Otherwise fallback already handled inside load_pubmed
        return texts

    print(f"  Using synthetic corpus ({len(SYNTHETIC_CORPUS)} sentences).")
    return SYNTHETIC_CORPUS


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(texts: list, min_len: int = 20, max_len: int = 512) -> list:
    """
    Clean and filter text sentences.

    Steps:
      1. Strip whitespace
      2. Remove lines with too many special characters (garbled text)
      3. Filter by length (too short = uninformative, too long = slow to train)
      4. Normalize whitespace
      5. Remove duplicates

    Args:
        texts  : raw list of strings
        min_len: minimum character length to keep
        max_len: maximum character length to keep

    Returns:
        cleaned, deduplicated list of strings
    """
    cleaned = []
    seen    = set()

    for text in texts:
        # 1. Basic strip
        text = text.strip()

        # 2. Skip too short or too long
        if len(text) < min_len or len(text) > max_len:
            continue

        # 3. Skip lines that are mostly non-alphabetic (garbled/latex/citations)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.6:
            continue

        # 4. Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # 5. Deduplicate
        if text.lower() in seen:
            continue
        seen.add(text.lower())

        cleaned.append(text)

    print(f"  Preprocessed: {len(texts)} → {len(cleaned)} sentences")
    return cleaned


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with synthetic first
    print("=== Synthetic corpus ===")
    corpus = preprocess(load_corpus(use_pubmed=False))
    print(f"  Final size: {len(corpus)}")
    print(f"  Sample: {corpus[0][:80]}...")

    # Test with PubMed
    print("\n=== PubMed corpus ===")
    corpus = preprocess(load_corpus(use_pubmed=True, pubmed_size=100))
    print(f"  Final size: {len(corpus)}")
    if corpus:
        print(f"  Sample: {corpus[0][:80]}...")
