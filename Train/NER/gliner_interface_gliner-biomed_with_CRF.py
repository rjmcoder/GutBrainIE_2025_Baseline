import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DebertaV2TokenizerFast
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from gliner import GLiNER
# from joblib import dump, load  # Uncomment if you want to save/load CRF models

# ----------------------------
# 1. Load gliner-biomed model
# ----------------------------
MODEL_PATH = "urchade/gliner_large-v2.1"  # replace with your path or HF repo ID
# MODEL_PATH = "Ihor/gliner-biomed-large-v1.0"

model = GLiNER.from_pretrained(
    MODEL_PATH,
    load_tokenizer=False,
    load_onnx_model=False
)

HF_BACKBONE = "microsoft/deberta-v2-xlarge"

# 2. Load its fast tokenizer
tokenizer = DebertaV2TokenizerFast.from_pretrained(HF_BACKBONE)

# 3. Attach to your GLiNER instance
model.tokenizer = tokenizer



# text = """
# The patient, a 45-year-old male, was diagnosed with type 2 diabetes mellitus and hypertension.
# He was prescribed Metformin 500mg twice daily and Lisinopril 10mg once daily. 
# A recent lab test showed elevated HbA1c levels at 8.2%.
# """

text = """
Parkinson's disease is a heterogeneous neurodegenerative disorder with distinctive gut microbiome patterns suggesting that interventions targeting the gut microbiota may prevent, slow, or reverse disease progression and severity. Because secretory IgA (SIgA) plays a key role in shaping the gut microbiota, characterization of the IgA-Biome of individuals classified into either the akinetic rigid (AR) or tremor dominant (TD) Parkinson's disease clinical subtypes was used to further define taxa unique to these distinct clinical phenotypes. Flow cytometry was used to separate IgA-coated and -uncoated bacteria from stool samples obtained from AR and TD patients followed by amplification and sequencing of the V4 region of the 16\u200aS rDNA gene on the MiSeq platform (Illumina). IgA-Biome analyses identified significant alpha and beta diversity differences between the Parkinson's disease phenotypes and the Firmicutes/Bacteroides ratio was significantly higher in those with TD compared to those with AR. In addition, discriminant taxa analyses identified a more pro-inflammatory bacterial profile in the IgA+ fraction of those with the AR clinical subclass compared to IgA-Biome analyses of those with the TD subclass and with the taxa identified in the unsorted control samples. IgA-Biome analyses underscores the importance of the host immune response in shaping the gut microbiome potentially affecting disease progression and presentation. In the present study, IgA-Biome analyses identified a unique proinflammatory microbial signature in the IgA+ fraction of those with AR that would have otherwise been undetected using conventional microbiome analysis approaches.
"""

labels = ["Anatomical Location", "Animal", "Biomedical Technique", "Bacteria", "Chemical", "Dietary Supplement", "Disease, Disorder, or Finding", "Drug",
          "Food", "Gene", "Human", "Microbiome", "Statistical Technique"]

entities = model.predict_entities(text, labels, threshold=0.5)

# for ent in entities:
#     print(f"{ent['text']} [{ent['label']}] ({ent['start']}-{ent['end']}) → {ent['score']:.2f}")


# Tokenize with offset mappings
encoding = tokenizer(
    text,
    return_tensors="pt",
    return_offsets_mapping=True,
    truncation=True,
    max_length=512
)

offsets = encoding["offset_mapping"].squeeze().tolist()
tokens  = tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())

bio_tags = ["O"] * len(tokens)

# assume `entities` is what you got from model.predict_entities(...)
for ent in entities:
    ent_start, ent_end = ent["start"], ent["end"]
    ent_label = ent["label"].upper().replace(" ", "_")

    # find all tokens whose char span overlaps
    tok_idxs = [
        i for i, (off_s, off_e) in enumerate(offsets)
        if not (off_e <= ent_start or off_s >= ent_end)
    ]
    if not tok_idxs:
        continue

    # First token = B-, rest = I-
    bio_tags[tok_idxs[0]] = f"B-{ent_label}"
    for idx in tok_idxs[1:]:
        bio_tags[idx] = f"I-{ent_label}"

# view BIO tags
for tok, tag in zip(tokens, bio_tags):
    print(f"{tok:<12} {tag}")


def extract_crf_features(tokens, bio_tags):
    features = []
    for i, tok in enumerate(tokens):
        feat = {
            "word.lower()":       tok.lower(),
            "word.isupper()":     tok.isupper(),
            "word.istitle()":     tok.istitle(),
            "word.isdigit()":     tok.isdigit(),
            "gliner_tag":         bio_tags[i],
        }
        # previous token’s tag
        if i > 0:
            feat["prev_gliner_tag"] = bio_tags[i-1]
        else:
            feat["BOS"] = True
        # next token’s tag
        if i < len(tokens)-1:
            feat["next_gliner_tag"] = bio_tags[i+1]
        else:
            feat["EOS"] = True
        features.append(feat)
    return features

print(extract_crf_features(tokens, bio_tags))