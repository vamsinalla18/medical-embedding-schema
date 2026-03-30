# data/kg_triples.py
# Knowledge Graph — 150+ entities, 100+ seed triples, corpus extraction

import re

TYPE_TO_ID = {"disease": 0, "drug": 1, "symptom": 2, "anatomy": 3}

SEED_ENTITY_TYPES = {
    # ── Diseases (40) ─────────────────────────────────────────────────────────
    "Diabetes":                 "disease",
    "Type2_Diabetes":           "disease",
    "Asthma":                   "disease",
    "Hypertension":             "disease",
    "Pneumonia":                "disease",
    "Cancer":                   "disease",
    "Lung_Cancer":              "disease",
    "Breast_Cancer":            "disease",
    "Obesity":                  "disease",
    "Alzheimer":                "disease",
    "Dementia":                 "disease",
    "Depression":               "disease",
    "Anxiety_Disorder":         "disease",
    "Arthritis":                "disease",
    "Rheumatoid_Arthritis":     "disease",
    "Osteoporosis":             "disease",
    "Tuberculosis":             "disease",
    "Malaria":                  "disease",
    "HIV":                      "disease",
    "Influenza":                "disease",
    "Stroke":                   "disease",
    "Epilepsy":                 "disease",
    "Anemia":                   "disease",
    "Bronchitis":               "disease",
    "Hepatitis":                "disease",
    "Hepatitis_B":              "disease",
    "Hepatitis_C":              "disease",
    "Psoriasis":                "disease",
    "Eczema":                   "disease",
    "Migraine":                 "disease",
    "Parkinson":                "disease",
    "Schizophrenia":            "disease",
    "Bipolar_Disorder":         "disease",
    "Kidney_Disease":           "disease",
    "Heart_Disease":            "disease",
    "Coronary_Artery_Disease":  "disease",
    "Heart_Failure":            "disease",
    "COPD":                     "disease",
    "Sepsis":                   "disease",
    "Appendicitis":             "disease",

    # ── Drugs (40) ────────────────────────────────────────────────────────────
    "Insulin":                  "drug",
    "Metformin":                "drug",
    "Aspirin":                  "drug",
    "Ibuprofen":                "drug",
    "Antibiotics":              "drug",
    "Beta_Blockers":            "drug",
    "Inhaler":                  "drug",
    "Paracetamol":              "drug",
    "Morphine":                 "drug",
    "Chemotherapy":             "drug",
    "Statins":                  "drug",
    "Antidepressants":          "drug",
    "Warfarin":                 "drug",
    "Vaccines":                 "drug",
    "Steroids":                 "drug",
    "Antivirals":               "drug",
    "Diuretics":                "drug",
    "Penicillin":               "drug",
    "Methotrexate":             "drug",
    "Lisinopril":               "drug",
    "Amoxicillin":              "drug",
    "Omeprazole":               "drug",
    "Atorvastatin":             "drug",
    "Amlodipine":               "drug",
    "Levothyroxine":            "drug",
    "Prednisone":               "drug",
    "Gabapentin":               "drug",
    "Sertraline":               "drug",
    "Fluoxetine":               "drug",
    "Ciprofloxacin":            "drug",
    "Azithromycin":             "drug",
    "Hydroxychloroquine":       "drug",
    "Remdesivir":               "drug",
    "Tamoxifen":                "drug",
    "Rituximab":                "drug",
    "ACE_Inhibitors":           "drug",
    "Calcium_Channel_Blockers": "drug",
    "Anticoagulants":           "drug",
    "Antifungals":              "drug",
    "Antihistamines":           "drug",

    # ── Symptoms (40) ─────────────────────────────────────────────────────────
    "Fever":                    "symptom",
    "Fatigue":                  "symptom",
    "Headache":                 "symptom",
    "Cough":                    "symptom",
    "Pain":                     "symptom",
    "Nausea":                   "symptom",
    "Vomiting":                 "symptom",
    "Dizziness":                "symptom",
    "Wheezing":                 "symptom",
    "Hyperglycemia":            "symptom",
    "Polyuria":                 "symptom",
    "Chills":                   "symptom",
    "Inflammation":             "symptom",
    "Bleeding":                 "symptom",
    "Swelling":                 "symptom",
    "Shortness_of_Breath":      "symptom",
    "Chest_Pain":               "symptom",
    "Weight_Loss":              "symptom",
    "Insomnia":                 "symptom",
    "Anxiety":                  "symptom",
    "Tremors":                  "symptom",
    "Seizures":                 "symptom",
    "Rash":                     "symptom",
    "Jaundice":                 "symptom",
    "Confusion":                "symptom",
    "Memory_Loss":              "symptom",
    "Joint_Pain":               "symptom",
    "Muscle_Weakness":          "symptom",
    "High_Blood_Pressure":      "symptom",
    "Low_Blood_Pressure":       "symptom",
    "Palpitations":             "symptom",
    "Numbness":                 "symptom",
    "Abdominal_Pain":           "symptom",
    "Diarrhea":                 "symptom",
    "Constipation":             "symptom",
    "Loss_of_Appetite":         "symptom",
    "Night_Sweats":             "symptom",
    "Frequent_Urination":       "symptom",
    "Blurred_Vision":           "symptom",
    "Sore_Throat":              "symptom",

    # ── Anatomy (30) ──────────────────────────────────────────────────────────
    "Heart":                    "anatomy",
    "Lungs":                    "anatomy",
    "Liver":                    "anatomy",
    "Kidney":                   "anatomy",
    "Brain":                    "anatomy",
    "Blood":                    "anatomy",
    "Blood_Sugar":              "anatomy",
    "Pancreas":                 "anatomy",
    "Immune_System":            "anatomy",
    "Nervous_System":           "anatomy",
    "Cardiovascular_System":    "anatomy",
    "Respiratory_System":       "anatomy",
    "Bone_Marrow":              "anatomy",
    "Lymph_Nodes":              "anatomy",
    "Thyroid":                  "anatomy",
    "Stomach":                  "anatomy",
    "Intestines":               "anatomy",
    "Spleen":                   "anatomy",
    "Skin":                     "anatomy",
    "Muscles":                  "anatomy",
    "Joints":                   "anatomy",
    "Arteries":                 "anatomy",
    "Veins":                    "anatomy",
    "Spine":                    "anatomy",
    "Eyes":                     "anatomy",
    "Ears":                     "anatomy",
    "Adrenal_Glands":           "anatomy",
    "Pituitary_Gland":          "anatomy",
    "Colon":                    "anatomy",
    "Bladder":                  "anatomy",
}


# ── Seed triples (100+) ────────────────────────────────────────────────────────
SEED_TRIPLES = [
    # Diabetes
    ("Diabetes",            "treated_by",   "Insulin"),
    ("Diabetes",            "treated_by",   "Metformin"),
    ("Diabetes",            "has_symptom",  "Hyperglycemia"),
    ("Diabetes",            "has_symptom",  "Fatigue"),
    ("Diabetes",            "has_symptom",  "Polyuria"),
    ("Diabetes",            "has_symptom",  "Frequent_Urination"),
    ("Diabetes",            "has_symptom",  "Blurred_Vision"),
    ("Diabetes",            "has_symptom",  "Weight_Loss"),
    ("Diabetes",            "affects",      "Pancreas"),
    ("Diabetes",            "affects",      "Blood_Sugar"),
    ("Diabetes",            "affects",      "Kidney"),
    ("Diabetes",            "affects",      "Eyes"),
    ("Type2_Diabetes",      "treated_by",   "Metformin"),
    ("Type2_Diabetes",      "has_symptom",  "Hyperglycemia"),
    ("Type2_Diabetes",      "has_symptom",  "Fatigue"),

    # Asthma
    ("Asthma",              "treated_by",   "Inhaler"),
    ("Asthma",              "treated_by",   "Steroids"),
    ("Asthma",              "has_symptom",  "Wheezing"),
    ("Asthma",              "has_symptom",  "Cough"),
    ("Asthma",              "has_symptom",  "Shortness_of_Breath"),
    ("Asthma",              "affects",      "Lungs"),
    ("Asthma",              "affects",      "Respiratory_System"),

    # Hypertension
    ("Hypertension",        "treated_by",   "Beta_Blockers"),
    ("Hypertension",        "treated_by",   "Lisinopril"),
    ("Hypertension",        "treated_by",   "ACE_Inhibitors"),
    ("Hypertension",        "treated_by",   "Calcium_Channel_Blockers"),
    ("Hypertension",        "treated_by",   "Amlodipine"),
    ("Hypertension",        "has_symptom",  "Headache"),
    ("Hypertension",        "has_symptom",  "Dizziness"),
    ("Hypertension",        "has_symptom",  "High_Blood_Pressure"),
    ("Hypertension",        "affects",      "Heart"),
    ("Hypertension",        "affects",      "Arteries"),
    ("Hypertension",        "affects",      "Kidney"),

    # Pneumonia
    ("Pneumonia",           "treated_by",   "Antibiotics"),
    ("Pneumonia",           "treated_by",   "Azithromycin"),
    ("Pneumonia",           "treated_by",   "Amoxicillin"),
    ("Pneumonia",           "has_symptom",  "Fever"),
    ("Pneumonia",           "has_symptom",  "Cough"),
    ("Pneumonia",           "has_symptom",  "Chest_Pain"),
    ("Pneumonia",           "has_symptom",  "Shortness_of_Breath"),
    ("Pneumonia",           "affects",      "Lungs"),

    # Cancer
    ("Cancer",              "treated_by",   "Chemotherapy"),
    ("Cancer",              "treated_by",   "Rituximab"),
    ("Cancer",              "has_symptom",  "Weight_Loss"),
    ("Cancer",              "has_symptom",  "Fatigue"),
    ("Cancer",              "has_symptom",  "Pain"),
    ("Cancer",              "affects",      "Immune_System"),
    ("Lung_Cancer",         "affects",      "Lungs"),
    ("Lung_Cancer",         "has_symptom",  "Cough"),
    ("Lung_Cancer",         "has_symptom",  "Shortness_of_Breath"),
    ("Breast_Cancer",       "treated_by",   "Tamoxifen"),
    ("Breast_Cancer",       "treated_by",   "Chemotherapy"),

    # Heart conditions
    ("Heart_Disease",       "has_symptom",  "Chest_Pain"),
    ("Heart_Disease",       "has_symptom",  "Shortness_of_Breath"),
    ("Heart_Disease",       "has_symptom",  "Palpitations"),
    ("Heart_Disease",       "affects",      "Heart"),
    ("Heart_Failure",       "treated_by",   "Diuretics"),
    ("Heart_Failure",       "treated_by",   "Beta_Blockers"),
    ("Heart_Failure",       "has_symptom",  "Swelling"),
    ("Heart_Failure",       "has_symptom",  "Fatigue"),
    ("Coronary_Artery_Disease", "affects",  "Arteries"),
    ("Coronary_Artery_Disease", "treated_by", "Statins"),
    ("Coronary_Artery_Disease", "treated_by", "Aspirin"),

    # Neurological
    ("Alzheimer",           "affects",      "Brain"),
    ("Alzheimer",           "has_symptom",  "Memory_Loss"),
    ("Alzheimer",           "has_symptom",  "Confusion"),
    ("Dementia",            "affects",      "Brain"),
    ("Dementia",            "has_symptom",  "Memory_Loss"),
    ("Parkinson",           "affects",      "Nervous_System"),
    ("Parkinson",           "has_symptom",  "Tremors"),
    ("Parkinson",           "has_symptom",  "Muscle_Weakness"),
    ("Epilepsy",            "affects",      "Nervous_System"),
    ("Epilepsy",            "has_symptom",  "Seizures"),
    ("Epilepsy",            "treated_by",   "Gabapentin"),
    ("Stroke",              "affects",      "Brain"),
    ("Stroke",              "has_symptom",  "Numbness"),
    ("Stroke",              "has_symptom",  "Confusion"),
    ("Migraine",            "has_symptom",  "Headache"),
    ("Migraine",            "has_symptom",  "Nausea"),
    ("Migraine",            "has_symptom",  "Blurred_Vision"),

    # Mental health
    ("Depression",          "treated_by",   "Antidepressants"),
    ("Depression",          "treated_by",   "Sertraline"),
    ("Depression",          "treated_by",   "Fluoxetine"),
    ("Depression",          "has_symptom",  "Insomnia"),
    ("Depression",          "has_symptom",  "Fatigue"),
    ("Depression",          "has_symptom",  "Loss_of_Appetite"),
    ("Anxiety_Disorder",    "has_symptom",  "Anxiety"),
    ("Anxiety_Disorder",    "has_symptom",  "Palpitations"),
    ("Anxiety_Disorder",    "has_symptom",  "Insomnia"),
    ("Schizophrenia",       "affects",      "Brain"),
    ("Bipolar_Disorder",    "treated_by",   "Antidepressants"),

    # Liver/GI
    ("Hepatitis",           "affects",      "Liver"),
    ("Hepatitis",           "has_symptom",  "Jaundice"),
    ("Hepatitis",           "has_symptom",  "Fatigue"),
    ("Hepatitis_B",         "treated_by",   "Antivirals"),
    ("Hepatitis_C",         "treated_by",   "Antivirals"),

    # Respiratory
    ("Tuberculosis",        "affects",      "Lungs"),
    ("Tuberculosis",        "has_symptom",  "Cough"),
    ("Tuberculosis",        "has_symptom",  "Night_Sweats"),
    ("Tuberculosis",        "has_symptom",  "Weight_Loss"),
    ("Bronchitis",          "affects",      "Lungs"),
    ("Bronchitis",          "has_symptom",  "Cough"),
    ("COPD",                "affects",      "Lungs"),
    ("COPD",                "affects",      "Respiratory_System"),
    ("COPD",                "has_symptom",  "Shortness_of_Breath"),
    ("COPD",                "treated_by",   "Inhaler"),

    # Musculoskeletal
    ("Arthritis",           "affects",      "Joints"),
    ("Arthritis",           "has_symptom",  "Joint_Pain"),
    ("Arthritis",           "has_symptom",  "Swelling"),
    ("Rheumatoid_Arthritis","treated_by",   "Methotrexate"),
    ("Rheumatoid_Arthritis","has_symptom",  "Joint_Pain"),
    ("Osteoporosis",        "affects",      "Spine"),
    ("Osteoporosis",        "has_symptom",  "Pain"),

    # Blood/immune
    ("Anemia",              "affects",      "Blood"),
    ("Anemia",              "has_symptom",  "Fatigue"),
    ("Anemia",              "has_symptom",  "Dizziness"),
    ("HIV",                 "affects",      "Immune_System"),
    ("HIV",                 "treated_by",   "Antivirals"),
    ("Malaria",             "has_symptom",  "Fever"),
    ("Malaria",             "has_symptom",  "Chills"),
    ("Influenza",           "treated_by",   "Antivirals"),
    ("Influenza",           "treated_by",   "Vaccines"),
    ("Influenza",           "has_symptom",  "Fever"),
    ("Influenza",           "has_symptom",  "Fatigue"),

    # Skin
    ("Psoriasis",           "affects",      "Skin"),
    ("Psoriasis",           "has_symptom",  "Rash"),
    ("Eczema",              "affects",      "Skin"),
    ("Eczema",              "has_symptom",  "Rash"),
    ("Eczema",              "has_symptom",  "Inflammation"),

    # Kidney
    ("Kidney_Disease",      "affects",      "Kidney"),
    ("Kidney_Disease",      "has_symptom",  "Swelling"),
    ("Kidney_Disease",      "has_symptom",  "Fatigue"),

    # Drug actions
    ("Insulin",             "regulates",    "Blood_Sugar"),
    ("Metformin",           "treats",       "Diabetes"),
    ("Aspirin",             "treats",       "Pain"),
    ("Aspirin",             "treats",       "Heart_Disease"),
    ("Ibuprofen",           "treats",       "Fever"),
    ("Ibuprofen",           "treats",       "Pain"),
    ("Statins",             "treats",       "Hypertension"),
    ("Statins",             "affects",      "Arteries"),
    ("Warfarin",            "affects",      "Blood"),
    ("Warfarin",            "treats",       "Heart_Disease"),
    ("Penicillin",          "treats",       "Pneumonia"),
    ("Antidepressants",     "treats",       "Depression"),
    ("Vaccines",            "treats",       "Influenza"),
    ("Steroids",            "treats",       "Inflammation"),
    ("Antihistamines",      "treats",       "Rash"),

    # Anatomy relations
    ("Heart",               "part_of",      "Cardiovascular_System"),
    ("Lungs",               "part_of",      "Respiratory_System"),
    ("Pancreas",            "part_of",      "Immune_System"),
    ("Brain",               "part_of",      "Nervous_System"),
    ("Bone_Marrow",         "part_of",      "Immune_System"),
    ("Colon",               "part_of",      "Intestines"),

    # Symptom chains
    ("Fever",               "has_symptom",  "Chills"),
    ("Fever",               "has_symptom",  "Fatigue"),
    ("Inflammation",        "has_symptom",  "Swelling"),
    ("Inflammation",        "has_symptom",  "Pain"),
    ("Obesity",             "has_symptom",  "Fatigue"),
    ("Obesity",             "affects",      "Heart"),
    ("Sepsis",              "has_symptom",  "Fever"),
    ("Sepsis",              "has_symptom",  "Confusion"),
    ("Sepsis",              "affects",      "Blood"),
]


# ── Relation patterns ──────────────────────────────────────────────────────────
RELATION_PATTERNS = {
    "treated_by": [
        r"(\w+)\s+(?:is|are|was)\s+treated\s+(?:by|with)\s+(\w+)",
        r"(\w+)\s+treatment\s+(?:includes?|uses?)\s+(\w+)",
        r"(\w+)\s+responds?\s+to\s+(\w+)",
        r"(\w+)\s+managed\s+with\s+(\w+)",
    ],
    "treats": [
        r"(\w+)\s+(?:treats?|cures?|manages?)\s+(\w+)",
        r"(\w+)\s+(?:is|are)\s+used\s+(?:to treat|for)\s+(\w+)",
        r"(\w+)\s+(?:reduces?|controls?|alleviates?)\s+(\w+)",
        r"(\w+)\s+prescribed\s+for\s+(\w+)",
    ],
    "has_symptom": [
        r"(\w+)\s+(?:causes?|presents?\s+with|characterized\s+by)\s+(\w+)",
        r"(\w+)\s+(?:symptoms?|signs?)\s+include\s+(\w+)",
        r"(\w+)\s+(?:is|are)\s+associated\s+with\s+(\w+)",
        r"patients?\s+with\s+(\w+)\s+(?:experience|report|develop)\s+(\w+)",
    ],
    "affects": [
        r"(\w+)\s+affects?\s+(?:the\s+)?(\w+)",
        r"(\w+)\s+(?:damages?|impairs?|injures?)\s+(?:the\s+)?(\w+)",
        r"(\w+)\s+(?:occurs?\s+in|found\s+in|targets?)\s+(?:the\s+)?(\w+)",
    ],
    "part_of": [
        r"(\w+)\s+(?:is|are)\s+part\s+of\s+(?:the\s+)?(\w+)",
        r"(\w+)\s+(?:belongs?\s+to|located\s+in|within)\s+(?:the\s+)?(\w+)",
    ],
    "regulates": [
        r"(\w+)\s+regulates?\s+(\w+)",
        r"(\w+)\s+controls?\s+(?:the\s+)?(\w+)\s+levels?",
        r"(\w+)\s+maintains?\s+(?:the\s+)?(\w+)",
    ],
}


# ── Triple extraction ─────────────────────────────────────────────────────────

def extract_triples_from_corpus(corpus: list, entity_types: dict) -> list:
    """Extract triples via regex + co-occurrence fallback."""
    entity_lookup = {}
    for name in entity_types:
        entity_lookup[name.replace("_", " ").lower()] = name
        entity_lookup[name.lower()] = name

    extracted = set()

    for sentence in corpus:
        s = sentence.lower()

        # Find mentioned entities
        mentioned = []
        seen = set()
        for surface, canonical in entity_lookup.items():
            if surface in s and canonical not in seen:
                mentioned.append(canonical)
                seen.add(canonical)

        if len(mentioned) < 2:
            continue

        # 1. Regex patterns
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
    """Build KG from seed + corpus extraction."""
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
        "Chemotherapy treats cancer and affects the immune system.",
        "Hypertension is associated with headache and affects the heart.",
        "Depression causes insomnia, fatigue and loss of appetite.",
        "Antibiotics treat pneumonia which affects the lungs.",
        "Stroke affects the brain and causes numbness and confusion.",
        "Asthma causes wheezing and shortness of breath.",
        "Parkinson disease causes tremors and muscle weakness.",
        "Hepatitis affects the liver and causes jaundice and fatigue.",
    ]
    triples, _, _, _, _, _ = build_kg(test_corpus)
    print(f"\nAfter extraction: {len(triples)} triples")
    for t in sorted(triples)[:15]:
        print(f"  {t}")
