import json

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
annotated_train_data = train_platinum + train_gold + train_silver
distant_train_data = train_bronze

json.dump(annotated_train_data, open("data/train_annotated.json", 'w', encoding='utf-8'))
print("Annotated training set for RE saved to data/train_annotated.json")

json.dump(distant_train_data, open("data/train_distant.json", 'w', encoding='utf-8'))
print("Distant training set for RE saved to data/train_distant.json")
