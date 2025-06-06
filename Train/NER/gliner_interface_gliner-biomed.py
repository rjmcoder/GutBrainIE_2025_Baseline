import json
from gliner import GLiNER

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoConfig
import os
import shutil
from types import SimpleNamespace
import types
import matplotlib.pyplot as plt

# set the current working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from importlib.metadata import version
version('GLiNER')

using_pretrained = False
finetune_model = False
generate_predictions = True

class DictNamespace(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dict = kwargs

    def __contains__(self, key):
        return key in self._dict

    def __getitem__(self, key):
        return self._dict[key]
    
# Set the GLiNER model to be used (from HuggingFace)
# model = GLiNER.from_pretrained("numind/NuNerZero")

tokenizer = AutoTokenizer.from_pretrained("./gliner-biomed-tokenizer")

# Add dummy tokens to pad tokenizer to size 128003
tokens_to_add = 128004 - len(tokenizer)
if tokens_to_add > 0:
    print(f"Padding tokenizer with {tokens_to_add} dummy tokens")
    new_tokens = [f"<dummy_{i}>" for i in range(tokens_to_add)]
    tokenizer.add_tokens(new_tokens)

# Save updated tokenizer
tokenizer.save_pretrained("./patched-tokenizer")

# # Load model
# model = GLiNER.from_pretrained(
#     "./gliner-biomed-cache/models--Ihor--gliner-biomed-base-v1.0/snapshots/9bfb24c62899ce6cb9e1a3b0dceb4b633926bfe5/",
#     tokenizer_name_or_path="./patched-tokenizer",
#     local_files_only=True
# )



# # Load tokenizer and patch
# tokenizer = AutoTokenizer.from_pretrained("./patched-tokenizer")
# print("Effective vocab size:", len(tokenizer))  # Should be 128003



# Now resize model's embedding layer
# model.token_rep_layer.bert_layer.model.resize_token_embeddings(len(tokenizer))

load_fine_tuned = True

if load_fine_tuned:

    # Load gliner_config.json manually
    with open("./outputs/gliner-biomed_finetuned_finetuned3_T60.0/gliner_config.json") as f:
        raw_config = json.load(f)

    # Wrap it
    gliner_config = DictNamespace(**raw_config)

    # Initialize model with config (weights NOT loaded yet)
    model = GLiNER(gliner_config)

    # Load raw model state dict from checkpoint
    checkpoint = torch.load("./outputs/gliner-biomed_finetuned_finetuned3_T60.0/pytorch_model.bin", map_location="cpu")

    # Remove the mismatching weight entry manually
    checkpoint.pop("token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight")

    # Load the rest of the model weights
    model.load_state_dict(checkpoint, strict=False)

    # Resize embedding layer BEFORE loading weights
    model.token_rep_layer.bert_layer.model.resize_token_embeddings(128004)

    model_name = "gliner-biomed_finetuned"

else:
    # Load gliner_config.json manually
    with open("./gliner-biomed-cache/models--Ihor--gliner-biomed-base-v1.0/snapshots/9bfb24c62899ce6cb9e1a3b0dceb4b633926bfe5/gliner_config.json") as f:
        raw_config = json.load(f)

    # Wrap it
    gliner_config = DictNamespace(**raw_config)

    # Initialize model with config (weights NOT loaded yet)
    model = GLiNER(gliner_config)

    # Load raw model state dict from checkpoint
    checkpoint = torch.load("./gliner-biomed-cache/models--Ihor--gliner-biomed-base-v1.0/snapshots/9bfb24c62899ce6cb9e1a3b0dceb4b633926bfe5/pytorch_model.bin", map_location="cpu")

    # Remove the mismatching weight entry manually
    checkpoint.pop("token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight")

    # Load the rest of the model weights
    model.load_state_dict(checkpoint, strict=False)

    # Resize embedding layer BEFORE loading weights
    model.token_rep_layer.bert_layer.model.resize_token_embeddings(128004)

    # Now load the model weights manually
    # model.load_state_dict(
    #     torch.load("./gliner-biomed-cache/models--Ihor--gliner-biomed-base-v1.0/snapshots/9bfb24c62899ce6cb9e1a3b0dceb4b633926bfe5/pytorch_model.bin", map_location="cpu")
    # )

    model_name = "gliner-biomed_finetuned"

# Define the confidence threshold to be used in evaluation
THRESHOLD = 0.6 


# Define the path to articles for which the final trained will generate predicted entities
PATH_ARTICLES = "../../Articles/json_format/articles_dev.json" 
PATH_OUTPUT_NER_PREDICTIONS = "../../Predictions/NER/gliner_biomed_finetuned3_predicted_entities.json"

print('## LOADING TRAINING DATA ##')
PATH_PLATINUM_TRAIN = "data/train_platinum.json"
PATH_GOLD_TRAIN = "data/train_gold.json"
PATH_SILVER_TRAIN = "data/train_silver.json"
PATH_BRONZE_TRAIN = "data/train_bronze.json"
PATH_DEV = "data/dev.json"

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
# Here we used the platinum, gold, and silver sets
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



# Define the hyperparameters in a config variable
config = SimpleNamespace(
    num_steps=5001, # 3000 # regulate number train, eval steps depending on the data size
    eval_every=250, #200,
    
    train_batch_size=4, # regulate batch size depending on GPU memory available.
    
    max_len=384, # maximum sentence length. 2048 for NuNerZero_long_context
    
    save_directory="logs", # log dir
    device='cuda' if torch.cuda.is_available() else 'cpu', #'cuda', # training device - cpu or cuda

    warmup_ratio=0.1, # Other parameters
    lr_encoder=1e-5,
    lr_others=5e-5,
    freeze_token_rep=False,

    max_types=15,
    shuffle_types=True,
    random_drop=True,
    max_neg_type_ratio=1,
)

# modify this to your own test data!
# don't forget to do the same preprocessing as for the train data:
# * converting entities-level data to token-level data
# * making entity_types lower-cased!!!
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

    # Initialize lists to track losses
    training_losses = []
    validation_losses = []

    epoch_loss = 0  # Accumulate loss for the epoch
    steps_in_epoch = len(train_loader)  # Number of steps in one epoch

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
            continue

        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients
        torch.cuda.empty_cache()        # added to clear GPU memory

        # Accumulate loss for the epoch
        epoch_loss += loss.item()

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
        pbar.set_description(description)

        if (step + 1) % steps_in_epoch == 0:  # End of an epoch
            avg_epoch_loss = epoch_loss / steps_in_epoch
            print(f"Epoch {step // steps_in_epoch} completed. Average Training Loss: {avg_epoch_loss:.4f}")
            training_losses.append(avg_epoch_loss)  # Track average loss for the epoch
            epoch_loss = 0  # Reset epoch loss accumulator

        if (step + 1) % steps_in_epoch == 0:  # End of an epoch

            model.eval()

            if eval_data is not None:
                results, f1 = model.evaluate(eval_data["samples"], flat_ner=True, threshold=THRESHOLD, batch_size=32,
                                     entity_types=eval_data["entity_types"])

                print(f"Step={step}\n{results}")

                # Calculate validation loss
                val_loss = 0
                for batch in model.create_dataloader(eval_data["samples"], batch_size=32, shuffle=False):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch[k] = v.to(config.device)
                    with torch.no_grad():
                        val_loss += model(batch).item()
                val_loss /= len(eval_data["samples"])
                validation_losses.append(val_loss)

                print(f"Validation Loss: {val_loss:.4f}")

            if not os.path.exists(config.save_directory):
                os.makedirs(config.save_directory)

            # save the intermediate best model
            if val_loss < min(validation_losses[:-1], default=float('inf')):
                print(f"Saving best model at step {step} with validation loss {val_loss:.4f}")
                if model_name == "gliner-biomed_finetuned":
                    model.config = recursive_namespace_to_dict(model.config)
                    output_path = f"{config.save_directory}/{model_name}_finetuned3_best"
                    model.save_pretrained(output_path)
                else:
                    model.save_pretrained(f"{config.save_directory}/finetuned_best")

            model.train()

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))

    # Plot training losses
    plt.plot(range(len(training_losses)), training_losses, label="Training Loss")

    # Plot validation losses
    # Use the correct x-axis values for validation losses
    validation_x = [i * (config.eval_every // steps_in_epoch) for i in range(len(validation_losses))]
    plt.plot(validation_x, validation_losses, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"{config.save_directory}/loss_plot3.png")
    plt.show()

output_path = f"outputs/{model_name}_finetuned3_T{str(THRESHOLD*100)}"

if finetune_model:
    print('## LAUNCHING TRAINING ##')
    train(model, config, train_data, eval_data)

    print('## SAVING TRAINED MODEL ##')
    
    
    if model_name == "gliner-biomed_finetuned":
        model.config = recursive_namespace_to_dict(model.config)

    model.save_pretrained(output_path)
    # os.system(f'cp {output_path}/gliner_config.json {output_path}/config.json')
    shutil.copy(f"{output_path}/gliner_config.json", f"{output_path}/config.json")
    md = GLiNER.from_pretrained(output_path, local_files_only=True)

if generate_predictions:
    # output_path = f"outputs/{model_name}_finetuned_T{str(THRESHOLD*100)}"
    print(f"## LOADING PRE-TRAINED MODEL {output_path} ##")

    md = model

    # if using_pretrained:
    #     md = model  # Load the pre-trained model as is
    # else:
    #     # output_path = f"outputs/{model_name}_finetuned_T{str(THRESHOLD*100)}"
    #     md = GLiNER.from_pretrained(output_path, local_files_only=True)

    print(f"## GENERATING NER PREDICTIONS FOR {PATH_ARTICLES}")
    with open(PATH_ARTICLES, 'r', encoding='utf-8') as file:
        articles = json.load(file)

    print(f"len(articles): {len(articles)}")
    entity_labels = eval_data['entity_types']

    # Dictionary to hold predicted entities
    # PMID -> {{'start_idx': ..., 'end_idx': ..., 'text_span': ..., 'entity_label': ..., 'score': ...}, ...}
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

        # Remove duplicates from title entities and add tag field with value 't'
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

        # Remove duplicates from abstract entities and add tag field with value 'a'
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

    