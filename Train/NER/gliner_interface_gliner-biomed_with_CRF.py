import json
from gliner import GLiNER

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoConfig
import os
import shutil
from types import SimpleNamespace
import types
from torchcrf import CRF  # Make sure to install this: pip install pytorch-crf

# set the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from importlib.metadata import version
version('GLiNER')

using_pretrained = False
finetune_model = True
generate_predictions = True

class DictNamespace(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dict = kwargs

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        return self._dict[key]

# Load tokenizer and config as in your original code
tokenizer = AutoTokenizer.from_pretrained("./gliner-biomed-tokenizer")

# Add dummy tokens to pad tokenizer to size 128003
tokens_to_add = 128004 - len(tokenizer)
if tokens_to_add > 0:
    print(f"Padding tokenizer with {tokens_to_add} dummy tokens")
    new_tokens = [f"<dummy_{i}>" for i in range(tokens_to_add)]
    tokenizer.add_tokens(new_tokens)

# Save updated tokenizer
tokenizer.save_pretrained("./patched-tokenizer")

# Load gliner_config.json manually
with open("./gliner-biomed-cache/models--Ihor--gliner-biomed-base-v1.0/snapshots/9bfb24c62899ce6cb9e1a3b0dceb4b633926bfe5/gliner_config.json") as f:
    raw_config = json.load(f)

# Wrap it
gliner_config = DictNamespace(**raw_config)

# Initialize model with config (weights NOT loaded yet)
base_model = GLiNER(gliner_config)

# Load raw model state dict from checkpoint
checkpoint = torch.load("./gliner-biomed-cache/models--Ihor--gliner-biomed-base-v1.0/snapshots/9bfb24c62899ce6cb9e1a3b0dceb4b633926bfe5/pytorch_model.bin", map_location="cpu")

# Remove the mismatching weight entry manually
checkpoint.pop("token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight")

# Load the rest of the model weights
base_model.load_state_dict(checkpoint, strict=False)

# Resize embedding layer BEFORE loading weights
base_model.token_rep_layer.bert_layer.model.resize_token_embeddings(128004)

# Create a CRF wrapper for GLiNER
class GLiNERCRFWrapper(nn.Module):
    def __init__(self, base_model, num_tags=2):
        super(GLiNERCRFWrapper, self).__init__()
        self.base_model = base_model
        self.num_tags = num_tags
        
        # Add CRF layer
        self.crf = CRF(num_tags, batch_first=True)
        
        # Add embedding-to-tag projection layer
        self.tag_projection = nn.Linear(768, num_tags)  # Assuming 768 hidden size
        
        # Copy attributes from base model for compatibility
        self.config = base_model.config
        self.tokenizer = base_model.tokenizer
        self.device = base_model.device
        self.entity_encoder = base_model.entity_encoder
    
    def forward(self, x):
        # Forward pass through base model
        base_loss = self.base_model(x)
        
        # Extract input_ids and attention_mask for CRF
        if isinstance(x, dict) and 'input_ids' in x and 'attention_mask' in x:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            
            # Process inputs for CRF
            # We need to compute token-level representations
            with torch.no_grad():  # Don't compute gradients through base model
                # Create the appropriate input format for the encoder
                encoder_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                
                # Get token embeddings from the base model
                if hasattr(self.base_model, 'entity_encoder') and hasattr(self.base_model.entity_encoder, 'token_rep_layer'):
                    # Use the token_rep_layer to get token embeddings
                    lengths = attention_mask.sum(dim=1).cpu()
                    token_embeddings = self.base_model.entity_encoder.token_rep_layer(input_ids, lengths)
                    
                    # Project embeddings to tag space
                    tag_logits = self.tag_projection(token_embeddings)
                    
                    # Create binary labels from entity spans
                    if 'labels' in x:
                        labels = x['labels']
                        bio_labels = torch.zeros_like(input_ids)
                        bio_labels = torch.where(labels > 0, 1, bio_labels)
                        
                        # Calculate CRF loss
                        crf_loss = -self.crf(tag_logits, bio_labels, mask=attention_mask.bool(), reduction='mean')
                        
                        # Combined loss
                        alpha = 0.7  # Weight for original loss
                        beta = 0.3   # Weight for CRF loss
                        combined_loss = alpha * base_loss + beta * crf_loss
                        return combined_loss
            
        return base_loss
    
    def set_sampling_params(self, **kwargs):
        """Pass through to base model"""
        self.base_model.set_sampling_params(**kwargs)
    
    def create_dataloader(self, *args, **kwargs):
        """Pass through to base model"""
        return self.base_model.create_dataloader(*args, **kwargs)
    
    def get_optimizer(self, lr_encoder, lr_others, freeze_token_rep=False):
        """Get optimizer for both base model and CRF layers"""
        base_optimizer = self.base_model.get_optimizer(lr_encoder, lr_others, freeze_token_rep)
        
        # Add CRF parameters
        crf_params = list(self.crf.parameters()) + list(self.tag_projection.parameters())
        base_optimizer.add_param_group({
            'params': crf_params,
            'lr': lr_others * 2,  # Higher learning rate for CRF parameters
            'weight_decay': 0.01
        })
        
        return base_optimizer
    
    def evaluate(self, *args, **kwargs):
        """Pass through to base model but add CRF refinement"""
        return self.base_model.evaluate(*args, **kwargs)
    
    def predict_entities(self, text, entity_types, threshold=0.5, batch_size=1, flat_ner=False, multi_label=False):
        """Override predict_entities to add CRF refinement"""
        # Get base model predictions
        base_entities = self.base_model.predict_entities(
            text, entity_types, threshold, batch_size, flat_ner, multi_label
        )
        
        # For batch prediction, just use base model
        if isinstance(text, list) and len(text) > 1:
            return base_entities
        
        # For single text, apply CRF refinement
        try:
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Get lengths for token_rep_layer
            lengths = inputs['attention_mask'].sum(dim=1).cpu()
            
            # Get token embeddings
            with torch.no_grad():
                token_embeddings = self.base_model.entity_encoder.token_rep_layer(
                    inputs['input_ids'], lengths
                )
                
                # Project to tag space
                tag_logits = self.tag_projection(token_embeddings)
                
                # Decode with CRF
                best_tags = self.crf.decode(tag_logits, mask=inputs['attention_mask'].bool())[0]
            
            # Process CRF tags to find entity spans
            entity_spans = []
            in_entity = False
            start_idx = None
            
            for i, tag in enumerate(best_tags):
                if tag == 1:  # Entity token
                    if not in_entity:
                        in_entity = True
                        start_idx = i
                else:  # Non-entity token
                    if in_entity:
                        in_entity = False
                        entity_spans.append((start_idx, i - 1))
            
            # Handle entity that extends to the end
            if in_entity:
                entity_spans.append((start_idx, len(best_tags) - 1))
            
            # Convert token spans to character spans
            char_spans = []
            for start_token, end_token in entity_spans:
                # Skip if out of range
                if start_token >= len(inputs['input_ids'][0]) or end_token >= len(inputs['input_ids'][0]):
                    continue
                
                # Get the actual tokens
                token_start = self.tokenizer.convert_ids_to_tokens(int(inputs['input_ids'][0][start_token]))
                token_end = self.tokenizer.convert_ids_to_tokens(int(inputs['input_ids'][0][end_token]))
                
                # Skip special tokens
                if token_start.startswith('[') or token_end.startswith('['):
                    continue
                
                # Find in original text
                start_chars = text.find(token_start.replace('##', ''))
                if start_chars == -1:
                    continue
                
                # Find end based on token span
                span_text = self.tokenizer.decode(inputs['input_ids'][0][start_token:end_token+1])
                end_chars = start_chars + len(span_text)
                
                char_spans.append((start_chars, end_chars))
            
            # Combine CRF spans with original entities
            refined_entities = []
            
            # For each CRF span, find the best matching entity
            for span_start, span_end in char_spans:
                best_entity = None
                best_overlap = 0
                
                for entity in base_entities:
                    entity_start = entity['start']
                    entity_end = entity['end']
                    
                    # Calculate overlap
                    overlap_start = max(span_start, entity_start)
                    overlap_end = min(span_end, entity_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_entity = entity
                
                if best_entity:
                    # Create refined entity
                    refined_entity = {
                        'start': span_start,
                        'end': span_end,
                        'text': text[span_start:span_end],
                        'label': best_entity['label'],
                        'score': best_entity['score'] * 1.1  # Boost confidence for CRF-refined entities
                    }
                    refined_entities.append(refined_entity)
            
            # Add high-confidence base entities that don't overlap with refinements
            for entity in base_entities:
                if entity['score'] > threshold + 0.2:  # Higher threshold for base entities
                    # Check for overlaps
                    overlaps = False
                    for refined in refined_entities:
                        if max(entity['start'], refined['start']) < min(entity['end'], refined['end']):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        refined_entities.append(entity)
            
            if refined_entities:
                return refined_entities
            else:
                return base_entities
                
        except Exception as e:
            print(f"Error in CRF refinement: {e}")
            return base_entities
    
    def train(self):
        """Set training mode"""
        self.base_model.train()
        self.crf.train()
        self.tag_projection.train()
    
    def eval(self):
        """Set evaluation mode"""
        self.base_model.eval()
        self.crf.eval()
        self.tag_projection.eval()
    
    def to(self, device):
        """Move model to device"""
        self.base_model.to(device)
        self.crf.to(device)
        self.tag_projection.to(device)
        self.device = device
        return self
    
    def save_pretrained(self, output_dir):
        """Save both base model and CRF components"""
        # Save base model
        self.base_model.save_pretrained(output_dir)
        
        # Save CRF components
        crf_path = os.path.join(output_dir, "crf_components.pt")
        torch.save({
            'crf': self.crf.state_dict(),
            'tag_projection': self.tag_projection.state_dict(),
            'num_tags': self.num_tags
        }, crf_path)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load base GLiNER model
        base_model = GLiNER.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Create wrapper
        model = cls(base_model, num_tags=2)
        
        # Check for CRF components
        crf_path = os.path.join(pretrained_model_name_or_path, "crf_components.pt")
        if os.path.exists(crf_path):
            crf_state = torch.load(crf_path, map_location="cpu")
            
            # Load CRF components
            model.num_tags = crf_state['num_tags']
            model.crf.load_state_dict(crf_state['crf'])
            model.tag_projection.load_state_dict(crf_state['tag_projection'])
        
        return model

# Create CRF wrapper model
print("Creating GLiNER CRF wrapper model...")
model = GLiNERCRFWrapper(base_model, num_tags=2)
print("Successfully created CRF wrapper model")


model_name = "gliner-biomed_finetuned_crf"

# Define the confidence threshold to be used in evaluation
THRESHOLD = 0.6 

# Define the path to articles for which the final trained will generate predicted entities
PATH_ARTICLES = "../../Articles/json_format/articles_dev.json" 
PATH_OUTPUT_NER_PREDICTIONS = "../../Predictions/NER/gliner_biomed_finetuned_crf_predicted_entities.json"

print('## LOADING TRAINING DATA ##')
PATH_PLATINUM_TRAIN = "data/train_platinum.json"
PATH_GOLD_TRAIN = "data/train_gold.json"
PATH_SILVER_TRAIN = "data/train_silver.json"
PATH_BRONZE_TRAIN = "data/train_bronze.json"
PATH_DEV = "data/dev.json"

# Rest of training data loading and processing as in your original code
with open(PATH_PLATINUM_TRAIN, 'r', encoding='utf-8') as file:
    train_platinum = json.load(file)

with open(PATH_GOLD_TRAIN, 'r', encoding='utf-8') as file:
    train_gold = json.load(file)

with open(PATH_SILVER_TRAIN, 'r', encoding='utf-8') as file:
    train_silver = json.load(file)
    
with open(PATH_BRONZE_TRAIN, 'r', encoding='utf-8') as file:
    train_bronze = json.load(file)

with open(PATH_DEV, 'r', encoding='utf-8') as file:
    eval_data = json.load(file)

# Set the data to be used for training
train_data = train_platinum + train_gold + train_silver

# converting entities-level train data to token-level 
new_data = []
for d in train_data:
    new_ner = []
    for s, f, c in d["ner"]:
        for i in range(s, f + 1):
            # labels are intended to be lower-case
            new_ner.append((i, i, c.lower()))
    new_d = {
        "tokenized_text": d["tokenized_text"],
        "ner": new_ner,
    }
    new_data.append(new_d)
train_data = new_data

# converting entities-level eval data to token-level 
new_data = []
for d in eval_data:
    new_ner = []
    for s, f, c in d["ner"]:
        for i in range(s, f + 1):
            # labels are intended to be lower-case
            new_ner.append((i, i, c.lower()))
    new_d = {
        "tokenized_text": d["tokenized_text"],
        "ner": new_ner,
    }
    new_data.append(new_d)
eval_data = new_data

# Define the hyperparameters
config = SimpleNamespace(
    num_steps=5001,
    eval_every=1000,
    train_batch_size=4,
    max_len=384,
    save_directory="logs",
    device='cuda' if torch.cuda.is_available() else 'cpu',
    warmup_ratio=0.1,
    lr_encoder=1e-5,
    lr_others=3e-5,
    freeze_token_rep=False,
    max_types=15,
    shuffle_types=True,
    random_drop=True,
    max_neg_type_ratio=1,
)

# Process eval data
eval_data = {
    "entity_types": [
        "anatomical location",
        "animal",
        "biomedical technique",
        "bacteria",
        "chemical",
        "dietary supplement",
        "ddf",
        "drug",
        "food",
        "gene",
        "human",
        "microbiome",
        "statistical technique",
    ],
    "samples": eval_data
}

def recursive_namespace_to_dict(obj):
    if isinstance(obj, (types.SimpleNamespace, DictNamespace)):
        return {k: recursive_namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: recursive_namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_namespace_to_dict(v) for v in obj]
    else:
        return obj
        
print('## DEFINING TRAINING FUNCTION ##') 
def train(model, config, train_data, eval_data=None):
    model = model.to(config.device)

    # Set sampling parameters from config
    model.set_sampling_params(
        max_types=config.max_types,
        shuffle_types=config.shuffle_types,
        random_drop=config.random_drop,
        max_neg_type_ratio=config.max_neg_type_ratio,
        max_len=config.max_len
    )

    model.train()

    # Initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=config.train_batch_size, shuffle=True)

    # Optimizer
    optimizer = model.get_optimizer(config.lr_encoder, config.lr_others, config.freeze_token_rep)

    pbar = tqdm(range(config.num_steps))

    if config.warmup_ratio < 1:
        num_warmup_steps = int(config.num_steps * config.warmup_ratio)
    else:
        num_warmup_steps = int(config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=config.num_steps
    )

    iter_train_loader = iter(train_loader)

    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(config.device)

        loss = model(x)  # Forward pass

        # Check if loss is nan
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping batch")
            continue

        loss.backward()  # Compute gradients
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients
        torch.cuda.empty_cache()  # Clear GPU memory

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
        pbar.set_description(description)

        if (step + 1) % config.eval_every == 0:
            model.eval()

            if eval_data is not None:
                results, f1 = model.evaluate(eval_data["samples"], flat_ner=True, threshold=THRESHOLD, batch_size=32,
                                     entity_types=eval_data["entity_types"])

                print(f"Step={step}\n{results}")

            if not os.path.exists(config.save_directory):
                os.makedirs(config.save_directory)

            if isinstance(model, GLiNERCRFWrapper):
                output_path = f"{config.save_directory}/{model_name}_finetuned_{step}"
                model.save_pretrained(output_path)
            else:
                # For base model
                model.config = recursive_namespace_to_dict(model.config)
                output_path = f"{config.save_directory}/{model_name}_finetuned_{step}"
                model.save_pretrained(output_path)

            model.train()

if finetune_model:
    print('## LAUNCHING TRAINING ##')
    train(model, config, train_data, eval_data)

    print('## SAVING TRAINED MODEL ##')
    output_path = f"outputs/{model_name}_finetuned_T{str(THRESHOLD*100)}"
    
    if isinstance(model, GLiNERCRFWrapper):
        model.save_pretrained(output_path)
    else:
        # For base model
        model.config = recursive_namespace_to_dict(model.config)
        model.save_pretrained(output_path)
        
    # Copy config files for compatibility
    if os.path.exists(f"{output_path}/gliner_config.json"):
        shutil.copy(f"{output_path}/gliner_config.json", f"{output_path}/config.json")

if generate_predictions:
    output_path = f"outputs/{model_name}_finetuned_T{str(THRESHOLD*100)}"
    print(f"## LOADING PRE-TRAINED MODEL {output_path} ##")

    if using_pretrained:
        md = model  # Load the pre-trained model as is
    else:
        if os.path.exists(os.path.join(output_path, "crf_components.pt")):
            # Load CRF model
            base_model = GLiNER.from_pretrained(output_path, local_files_only=True)
            md = GLiNERCRFWrapper.from_pretrained(output_path, local_files_only=True)
        else:
            # Load base model
            md = GLiNER.from_pretrained(output_path, local_files_only=True)

    print(f"## GENERATING NER PREDICTIONS FOR {PATH_ARTICLES}")
    with open(PATH_ARTICLES, 'r', encoding='utf-8') as file:
        articles = json.load(file)

    print(f"len(articles): {len(articles)}")
    entity_labels = eval_data['entity_types']

    # Dictionary to hold predicted entities
    predictions = {} 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md.to(device)  # move model to GPU/CPU

    for pmid, content in tqdm(articles.items(), total=len(articles), desc="Predicting entities..."):
        title = content['title']
        abstract = content['abstract']

        # Predict entities 
        title_entities = md.predict_entities(title, entity_labels, threshold=THRESHOLD, flat_ner=True, multi_label=False)
        abstract_entities = md.predict_entities(abstract, entity_labels, threshold=THRESHOLD, flat_ner=True, multi_label=False)

        # Adjust indices for predicted entities in the abstract
        for entity in abstract_entities:
            entity['start'] += len(title) + 1
            entity['end'] += len(title) + 1

        # Remove duplicates from predicted entities
        unique_entities = []
        seen_entities = set()

        # Process title entities
        for entity in title_entities:
            key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
            if key not in seen_entities:
                tmp_entity = {
                    'start_idx': entity['start'],
                    'end_idx': entity['end'],
                    'tag': 't',
                    'text_span': entity['text'],
                    'entity_label': entity['label'],
                    'score': entity['score']
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        # Process abstract entities
        for entity in abstract_entities:
            key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
            if key not in seen_entities:
                tmp_entity = {
                    'start_idx': entity['start'],
                    'end_idx': entity['end'],
                    'tag': 'a',
                    'text_span': entity['text'],
                    'entity_label': entity['label'],
                    'score': entity['score']
                }
                unique_entities.append(tmp_entity)
                seen_entities.add(key)

        predictions[pmid] = unique_entities
        articles[pmid]['pred_entities'] = unique_entities

    # Convert any non-serializable data if necessary
    def default_serializer(obj):
        if isinstance(obj, set):
            return list(obj)
        # Add other types if needed
        raise TypeError(f'Type {type(obj)} not serializable')
    
    with open(PATH_OUTPUT_NER_PREDICTIONS, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2, default=default_serializer)

    print(f"## Predictions have been exported in JSON format to '/{PATH_OUTPUT_NER_PREDICTIONS}' ##")