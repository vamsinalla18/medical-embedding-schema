# data/corpus.py
# Medical text corpus — loads from multiple HuggingFace datasets
# Supports up to 50k samples with robust fallback chain

import re


# ── Synthetic fallback ────────────────────────────────────────────────────────
SYNTHETIC_CORPUS = [
    "Diabetes is a chronic metabolic disease characterized by elevated blood sugar levels.",
    "Type 2 diabetes is commonly treated with metformin and insulin therapy.",
    "Insulin is a hormone produced by the pancreas that regulates blood sugar.",
    "Hyperglycemia, or high blood sugar, is the primary symptom of uncontrolled diabetes.",
    "Fatigue and polyuria are common early symptoms of diabetes mellitus.",
    "Asthma is a chronic respiratory disease that causes inflammation of the airways.",
    "Patients with asthma often use inhalers to manage wheezing and breathing difficulties.",
    "Hypertension, or high blood pressure, is a major risk factor for cardiovascular disease.",
    "Beta blockers are commonly prescribed to treat hypertension and heart conditions.",
    "Pneumonia is a serious lung infection that causes fever, cough, and chest pain.",
    "Bacterial pneumonia is typically treated with a course of antibiotics.",
    "The cardiovascular system includes the heart and blood vessels.",
    "Blood sugar regulation is critical for metabolic homeostasis.",
    "Ibuprofen is a nonsteroidal anti-inflammatory drug used to reduce fever and pain.",
    "Medical knowledge graphs capture structured relationships between diseases, drugs, and symptoms.",
]


# ── Individual dataset loaders ────────────────────────────────────────────────

def _load_medalpaca(num_samples: int) -> list:
    """medalpaca/medical_meadow_pubmed_causal — cause-effect PubMed sentences."""
    from datasets import load_dataset
    print("    Loading medalpaca/medical_meadow_pubmed_causal...")
    ds = load_dataset("medalpaca/medical_meadow_pubmed_causal", split="train")
    texts = []
    for item in ds:
        for field in ["input", "output", "instruction"]:
            val = item.get(field, "")
            if isinstance(val, str) and len(val.strip()) > 30:
                texts.append(val.strip())
        if len(texts) >= num_samples:
            break
    print(f"    → {len(texts)} samples")
    return texts[:num_samples]


def _load_pubmed_qa(num_samples: int) -> list:
    """pubmed_qa — real PubMed abstracts with context paragraphs."""
    from datasets import load_dataset
    print("    Loading pubmed_qa pqa_labeled...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    texts = []
    for item in ds:
        ctx = item.get("context", {})
        if isinstance(ctx, dict):
            for para in ctx.get("contexts", []):
                if isinstance(para, str) and len(para.strip()) > 30:
                    texts.append(para.strip())
        if len(texts) >= num_samples:
            break
    print(f"    → {len(texts)} samples")
    return texts[:num_samples]


def _load_medical_questions(num_samples: int) -> list:
    """lavita/medical-qa-datasets — medical QA pairs."""
    from datasets import load_dataset
    print("    Loading lavita/medical-qa-datasets...")
    ds = load_dataset("lavita/medical-qa-datasets", "all-processed", split="train")
    texts = []
    for item in ds:
        for field in ["input", "output", "instruction", "question", "answer"]:
            val = item.get(field, "")
            if isinstance(val, str) and len(val.strip()) > 30:
                texts.append(val.strip())
        if len(texts) >= num_samples:
            break
    print(f"    → {len(texts)} samples")
    return texts[:num_samples]


def _load_medical_dialog(num_samples: int) -> list:
    """medical_dialog — doctor-patient conversations."""
    from datasets import load_dataset
    print("    Loading medical_dialog...")
    ds = load_dataset("medical_dialog", "processed.en", split="train")
    texts = []
    for item in ds:
        utterances = item.get("utterances", [])
        for utt in utterances:
            if isinstance(utt, str) and len(utt.strip()) > 30:
                texts.append(utt.strip())
        if len(texts) >= num_samples:
            break
    print(f"    → {len(texts)} samples")
    return texts[:num_samples]


def _load_symptom_disease(num_samples: int) -> list:
    """
    gretelai/symptom_to_diagnosis — symptom descriptions mapped to diseases.
    Very relevant for symptom checker downstream task.
    """
    from datasets import load_dataset
    print("    Loading gretelai/symptom_to_diagnosis...")
    ds = load_dataset("gretelai/symptom_to_diagnosis", split="train")
    texts = []
    for item in ds:
        for field in ["input_text", "label_text", "text", "symptoms", "diagnosis"]:
            val = item.get(field, "")
            if isinstance(val, str) and len(val.strip()) > 10:
                texts.append(val.strip())
        if len(texts) >= num_samples:
            break
    print(f"    → {len(texts)} samples")
    return texts[:num_samples]


# ── Main multi-source loader ───────────────────────────────────────────────────

def load_pubmed(num_samples: int = 50000) -> list:
    """
    Load medical text from multiple HuggingFace datasets.
    Falls back gracefully if any dataset fails.

    Priority order:
      1. medalpaca (most reliable, already cached)
      2. pubmed_qa (real abstracts)
      3. gretelai/symptom_to_diagnosis (great for symptom checker)
      4. lavita/medical-qa-datasets
      5. medical_dialog
    """
    all_texts  = []
    per_source = num_samples // 3   # spread across sources

    loaders = [
        (_load_medalpaca,         per_source),
        (_load_symptom_disease,   per_source),
        (_load_pubmed_qa,         per_source),
        (_load_medical_questions, per_source // 2),
    ]

    for loader_fn, n in loaders:
        if len(all_texts) >= num_samples:
            break
        try:
            texts = loader_fn(n)
            all_texts.extend(texts)
            print(f"    Running total: {len(all_texts)}")
        except Exception as e:
            print(f"    Skipped ({loader_fn.__name__}): {e}")

    if len(all_texts) < 100:
        print("  All loaders failed — using synthetic corpus.")
        return SYNTHETIC_CORPUS

    print(f"  Total raw samples collected: {len(all_texts)}")
    return all_texts[:num_samples]


def load_corpus(use_pubmed: bool = True, pubmed_size: int = 50000) -> list:
    """
    Load the medical text corpus.

    Args:
        use_pubmed : load real data from HuggingFace (requires internet)
        pubmed_size: target number of samples (default 50000)
    """
    if use_pubmed:
        return load_pubmed(pubmed_size)
    print(f"  Using synthetic corpus ({len(SYNTHETIC_CORPUS)} sentences).")
    return SYNTHETIC_CORPUS


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(texts: list, min_len: int = 20, max_len: int = 512) -> list:
    """
    Clean and filter sentences.

    Steps:
      1. Strip whitespace
      2. Filter by length
      3. Skip garbled text (low alpha ratio)
      4. Normalize whitespace
      5. Deduplicate
    """
    cleaned = []
    seen    = set()

    for text in texts:
        text = text.strip()

        if len(text) < min_len or len(text) > max_len:
            continue

        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.55:
            continue

        text = re.sub(r"\s+", " ", text)

        key = text.lower()
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(text)

    print(f"  Preprocessed: {len(texts)} → {len(cleaned)} sentences")
    return cleaned


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    corpus = preprocess(load_corpus(use_pubmed=True, pubmed_size=1000))
    print(f"Final size: {len(corpus)}")
    print(f"Sample 1: {corpus[0][:100]}")
    print(f"Sample 2: {corpus[1][:100]}")
