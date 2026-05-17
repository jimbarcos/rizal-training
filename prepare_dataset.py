from datasets import load_dataset, DatasetDict
from transformers import MarianTokenizer

# 1. Load your newly created dataset
print("Loading dataset...")
dataset = load_dataset('csv', data_files='rizal_aligned_dataset.csv')

# 2. Split into Train (90%) and Test (10%) sets
print("Splitting into train/test...")
train_test = dataset['train'].train_test_split(test_size=0.1, seed=42)
dataset_dict = DatasetDict({
    'train': train_test['train'],
    'test': train_test['test']
})

# 3. Initialize the Marian Tokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-tl-en"
tokenizer = MarianTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    # --- CHOOSE YOUR INPUT ---
    #["original_tagalog"] or ["normalized_tagalog"] 
    inputs = examples["normalized_tagalog"] 
    
    # The target output is always English
    targets = examples["english"]
    
    # Pass inputs and targets to the tokenizer
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    
    return model_inputs

# 4. Apply tokenization across the dataset
print("Tokenizing data...")
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

# 5. Save the tokenized format to disk
tokenized_datasets.save_to_disk("rizal_tokenized_dataset")
print("Dataset successfully tokenized and saved to 'rizal_tokenized_dataset' directory!")