# data/kg_triples.py
# Knowledge Graph triples — expanded seed KG + auto-extracted from corpus

import re

TYPE_TO_ID = {"disease": 0, "drug": 1, "symptom": 2, "anatomy": 3}

SEED_ENTITY_TYPES = {
    # Diseases
    "Diabetes":             "disease",
    "Asthma":               "disease",
    "Hypertension":         "disease",
    "Pneumonia":            "disease",
    "Cancer":               "disease",
    "Obesity":              "disease",
    "Alzheimer":            "disease",
    "Depression":           "disease",
    "Arthritis":            "disease",
    "Osteoporosis":         "disease",
    "Tuberculosis":         "disease",
    "Malaria":              "disease",
    "HIV":                  "disease",
    "Influenza":            "disease",
    "Stroke":               "disease",
    "Epilepsy":             "disease",
    "Anemia":               "disease",
    "Bronchitis":           "disease",
    "Hepatitis":            "disease",
    "Psoriasis":            "disease",
    # Drugs
    "Insulin":              "drug",
    "Metformin":            "drug",
    "Aspirin":              "drug",
    "Ibuprofen":            "drug",
    "Antibiotics":          "drug",
    "Beta_Blockers":        "drug",
    "Inhaler":              "drug",
    "Paracetamol":          "drug",
    "Morphine":             "drug",
    "Chemotherapy":         "drug",
    "Statins":              "drug",
    "Antidepressants":      "drug",
    "Warfarin":             "drug",
    "Vaccines":             "drug",
    "Steroids":             "drug",
    "Antivirals":           "drug",
    "Diuretics":            "drug",
    "Penicillin":           "drug",
    "Methotrexate":         "drug",
    "Lisinopril":           "drug",
    # Symptoms
    "Fever":                "symptom",
    "Fatigue":              "symptom",
    "Headache":             "symptom",
    "Cough":                "symptom",
    "Pain":                 "symptom",
    "Nausea":               "symptom",
    "Vomiting":             "symptom",
    "Dizziness":            "symptom",
    "Wheezing":             "symptom",
    "Hyperglycemia":        "symptom",
    "Polyuria":             "symptom",
    "Chills":               "symptom",
    "Inflammation":         "symptom",
    "Bleeding":             "symptom",
    "Swelling":             "symptom",
    "Shortness_of_Breath":  "symptom",
    "Chest_Pain":           "symptom",
    "Weight_Loss":          "symptom",
    "Insomnia":             "symptom",
    "Anxiety":              "symptom",
    # Anatomy
    "Heart":                "anatomy",
    "Lungs":                "anatomy",
    "Liver":                "anatomy",
    "Kidney":               "anatomy",
    "Brain":                "anatomy",
    "Blood":                "anatomy",
    "Blood_Sugar":          "anatomy",
    "Pancreas":             "anatomy",
    "Immune_System":        "anatomy",
    "Nervous_System":       "anatomy",
    "Cardiovascular_System":"anatomy",
    "Respiratory_System":   "anatomy",
    "Bone_Marrow":          "anatomy",
    "Lymph_Nodes":          "anatomy",
    "Thyroid":              "anatomy",
}

SEED_TRIPLES = [
    ("Diabetes",        "treated_by",   "Insulin"),
    ("Diabetes",        "treated_by",   "Metformin"),
    ("Diabetes",        "has_symptom",  "Hyperglycemia"),
    ("Diabetes",        "has_symptom",  "Fatigue"),
    ("Diabetes",        "has_symptom",  "Polyuria"),
    ("Diabetes",        "affects",      "Pancreas"),
    ("Diabetes",        "affects",      "Blood_Sugar"),
    ("Asthma",          "treated_by",   "Inhaler"),
    ("Asthma",          "has_symptom",  "Wheezing"),
    ("Asthma",          "has_symptom",  "Cough"),
    ("Asthma",          "affects",      "Lungs"),
    ("Hypertension",    "treated_by",   "Beta_Blockers"),
    ("Hypertension",    "treated_by",   "Lisinopril"),
    ("Hypertension",    "has_symptom",  "Headache"),
    ("Hypertension",    "has_symptom",  "Dizziness"),
    ("Hypertension",    "affects",      "Heart"),
    ("Pneumonia",       "treated_by",   "Antibiotics"),
    ("Pneumonia",       "has_symptom",  "Fever"),
    ("Pneumonia",       "has_symptom",  "Cough"),
    ("Pneumonia",       "affects",      "Lungs"),
    ("Cancer",          "treated_by",   "Chemotherapy"),
    ("Cancer",          "has_symptom",  "Weight_Loss"),
    ("Cancer",          "has_symptom",  "Fatigue"),
    ("Cancer",          "affects",      "Immune_System"),
    ("Heart",           "part_of",      "Cardiovascular_System"),
    ("Lungs",           "part_of",      "Respiratory_System"),
    ("Pancreas",        "part_of",      "Immune_System"),
    ("Insulin",         "regulates",    "Blood_Sugar"),
    ("Metformin",       "treats",       "Diabetes"),
    ("Aspirin",         "treats",       "Pain"),
    ("Ibuprofen",       "treats",       "Fever"),
    ("Ibuprofen",       "treats",       "Pain"),
    ("Statins",         "treats",       "Hypertension"),
    ("Warfarin",        "affects",      "Blood"),
    ("Penicillin",      "treats",       "Pneumonia"),
    ("Antidepressants", "treats",       "Depression"),
    ("Vaccines",        "treats",       "Influenza"),
    ("Fever",           "has_symptom",  "Chills"),
    ("Inflammation",    "has_symptom",  "Swelling"),
    ("Anemia",          "affects",      "Blood"),
    ("Stroke",          "affects",      "Brain"),
    ("Alzheimer",       "affects",      "Brain"),
    ("Epilepsy",        "affects",      "Nervous_System"),
    ("Hepatitis",       "affects",      "Liver"),
    ("Tuberculosis",    "affects",      "Lungs"),
    ("Obesity",         "has_symptom",  "Fatigue"),
    ("Depression",      "has_symptom",  "Insomnia"),
    ("Depression",      "has_symptom",  "Anxiety"),
]

RELATION_PATTERNS = {
    "treated_by": [
        r"(\w+)\s+(?:is|are|was)\s+treated\s+(?:by|with)\s+(\w+)",
        r"(\w+)\s+treatment\s+(?:includes?|uses?)\s+(\w+)",
        r"(\w+)\s+responds?\s+to\s+(\w+)",
    ],
    "treats": [
        r"(\w+)\s+(?:treats?|cures?|manages?)\s+(\w+)",
        r"(\w+)\s+(?:is|are)\s+used\s+(?:to treat|for)\s+(\w+)",
        r"(\w+)\s+(?:reduces?|controls?)\s+(\w+)",
    ],
    "has_symptom": [
        r"(\w+)\s+(?:causes?|presents?\s+with|characterized\s+by)\s+(\w+)",
        r"(\w+)\s+(?:symptoms?|signs?)\s+include\s+(\w+)",
        r"(\w+)\s+(?:is|are)\s+associated\s+with\s+(\w+)",
    ],
    "affects": [
        r"(\w+)\s+affects?\s+(?:the\s+)?(\w+)",
        r"(\w+)\s+(?:damages?|impairs?)\s+(?:the\s+)?(\w+)",
        r"(\w+)\s+(?:occurs?\s+in|found\s+in)\s+(?:the\s+)?(\w+)",
    ],
    "part_of": [
        r"(\w+)\s+(?:is|are)\s+part\s+of\s+(?:the\s+)?(\w+)",
        r"(\w+)\s+(?:belongs?\s+to|located\s+in)\s+(?:the\s+)?(\w+)",
    ],
    "regulates": [
        r"(\w+)\s+regulates?\s+(\w+)",
        r"(\w+)\s+controls?\s+(?:the\s+)?(\w+)\s+levels?",
    ],
}


def extract_triples_from_corpus(corpus: list, entity_types: dict) -> list:
    """
    Extract triples from corpus via regex patterns + co-occurrence fallback.
    Co-occurrence covers ALL meaningful entity type-pair combinations.
    """
    # Build lookup: lowercase surface → canonical name
    entity_lookup = {}
    for name in entity_types:
        entity_lookup[name.replace("_", " ").lower()] = name
        entity_lookup[name.lower()] = name

    extracted = set()

    for sentence in corpus:
        s = sentence.lower()

        # Find all entities mentioned in this sentence (deduplicated)
        mentioned = []
        seen = set()
        for surface, canonical in entity_lookup.items():
            if surface in s and canonical not in seen:
                mentioned.append(canonical)
                seen.add(canonical)

        if len(mentioned) < 2:
            continue

        # 1. Regex pattern matching
        for relation, patterns in RELATION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, s):
                    w1, w2 = match.group(1), match.group(2)
                    head = entity_lookup.get(w1)
                    tail = entity_lookup.get(w2)
                    if head and tail and head != tail:
                        extracted.add((head, relation, tail))

        # 2. Co-occurrence fallback — all type-pair combinations
        for i in range(len(mentioned)):
            for j in range(i + 1, len(mentioned)):
                e1, e2 = mentioned[i], mentioned[j]
                t1, t2 = entity_types.get(e1), entity_types.get(e2)

                if t1 == "disease" and t2 == "drug":
                    extracted.add((e1, "treated_by", e2))
                elif t1 == "drug" and t2 == "disease":
                    extracted.add((e2, "treated_by", e1))

                elif t1 == "disease" and t2 == "symptom":
                    extracted.add((e1, "has_symptom", e2))
                elif t1 == "symptom" and t2 == "disease":
                    extracted.add((e2, "has_symptom", e1))

                elif t1 == "disease" and t2 == "anatomy":
                    extracted.add((e1, "affects", e2))
                elif t1 == "anatomy" and t2 == "disease":
                    extracted.add((e2, "affects", e1))

                elif t1 == "drug" and t2 == "symptom":
                    extracted.add((e1, "treats", e2))
                elif t1 == "symptom" and t2 == "drug":
                    extracted.add((e2, "treats", e1))

                elif t1 == "drug" and t2 == "anatomy":
                    extracted.add((e1, "affects", e2))
                elif t1 == "anatomy" and t2 == "drug":
                    extracted.add((e2, "affects", e1))

                elif t1 == "symptom" and t2 == "anatomy":
                    extracted.add((e1, "affects", e2))
                elif t1 == "anatomy" and t2 == "symptom":
                    extracted.add((e2, "affects", e1))

    return list(extracted)


def build_kg(corpus: list = None):
    """Build KG from seed triples + corpus extraction."""
    entity_types = dict(SEED_ENTITY_TYPES)
    triples      = set(map(tuple, SEED_TRIPLES))

    if corpus:
        print("  [KG] Extracting triples from corpus...")
        extracted = extract_triples_from_corpus(corpus, entity_types)
        valid = [(h, r, t) for h, r, t in extracted
                 if h in entity_types and t in entity_types]
        triples.update(valid)
        print(f"  [KG] Seed: {len(SEED_TRIPLES)} | "
              f"Extracted: {len(valid)} | Total: {len(triples)}")
    else:
        print(f"  [KG] Seed triples only: {len(triples)}")

    triples        = list(triples)
    entities       = list(entity_types.keys())
    relations      = list(set(r for _, r, _ in triples))
    entity_to_id   = {e: i for i, e in enumerate(entities)}
    relation_to_id = {r: i for i, r in enumerate(relations)}

    return triples, entity_types, entities, relations, entity_to_id, relation_to_id


# Module-level exports
ENTITY_TYPES   = dict(SEED_ENTITY_TYPES)
TRIPLES        = list(SEED_TRIPLES)
ENTITIES       = list(ENTITY_TYPES.keys())
RELATIONS      = list(set(r for _, r, _ in TRIPLES))
ENTITY_TO_ID   = {e: i for i, e in enumerate(ENTITIES)}
RELATION_TO_ID = {r: i for i, r in enumerate(RELATIONS)}


if __name__ == "__main__":
    from collections import Counter
    print(f"Seed entities : {len(ENTITY_TYPES)}")
    print(f"Seed triples  : {len(TRIPLES)}")
    print(f"Type counts:")
    for t, c in Counter(ENTITY_TYPES.values()).items():
        print(f"  {t:10s}: {c}")

    test_corpus = [
        "Metformin is used to treat diabetes and reduces blood sugar levels.",
        "Insulin regulates blood sugar in patients with diabetes.",
        "Aspirin treats pain and fever effectively.",
        "Chemotherapy is used to treat cancer and affects the immune system.",
        "Hypertension is associated with headache and dizziness.",
        "Depression causes insomnia and anxiety in many patients.",
        "Antibiotics treat pneumonia which affects the lungs.",
        "Warfarin affects blood clotting and is used in stroke patients.",
    ]
    triples, _, _, _, _, _ = build_kg(test_corpus)
    print(f"\nAfter extraction: {len(triples)} triples")
    for t in sorted(triples)[:15]:
        print(f"  {t}")
