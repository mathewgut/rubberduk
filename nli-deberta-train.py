from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set CUDA device to -1 to disable GPU usage

# This code uses the cpu to train a model based on the online terms of service data set on hugging face. With my 9900k running at 4.8ghz it takes approximatley 32~ hours


# Step 1: Load the dataset
dataset = load_dataset('joelito/online_terms_of_service', split='train')

# Step 2: Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-base')
label_column_name = 'unfairness_level'  # Replace with the correct column name

# Convert the column to a list if it is a single-column Dataset
if isinstance(dataset[label_column_name], list):
    labels = dataset[label_column_name]
else:
    labels = dataset[label_column_name].tolist()

label_values = list(set(labels))
num_labels = len(label_values)
label_encoder = {label: index for index, label in enumerate(label_values)}
mlb = MultiLabelBinarizer()
mlb.fit([[label_encoder[label]] for label in label_values])  # Fit the MultiLabelBinarizer

# Load the model with `ignore_mismatched_sizes` set to True
model = AutoModelForSequenceClassification.from_pretrained(
    'cross-encoder/nli-deberta-base',
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# Step 3: Define the preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(
        examples['sentence'],  # Replace 'text' with the correct column name
        truncation=True,
        padding='max_length',
        max_length=512  # Set the maximum sequence length
    )
    examples['input_ids'] = inputs.input_ids
    examples['attention_mask'] = inputs.attention_mask
    examples['labels'] = mlb.transform([[label_encoder[label]] for label in examples["unfairness_level"]])
    return examples

# Step 4: Preprocess the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Step 5: Define the compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_score': f1_score(labels, preds, average='weighted')
    }

# Step 6: Create and train the Trainer
training_args = TrainingArguments(
    output_dir='./finetune_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./finetune_logs',
)

# Force training on CPU
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
