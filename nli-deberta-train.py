from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
# Step 1: Load the dataset
dataset = load_dataset('joelito/online_terms_of_service')
# Split the dataset into training and validation subsets
train_dataset = dataset['train']
eval_dataset = dataset['validation']
# Step 2: Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
label_column_name = 'unfairness_level'  # Replace with the correct column name
# Convert the column to a list if it is a single-column Dataset
if isinstance(train_dataset[label_column_name], list):
    labels = train_dataset[label_column_name]
else:
    labels = train_dataset[label_column_name].tolist()
label_values = list(set(labels))
num_labels = len(label_values)
label_encoder = {label: index for index, label in enumerate(label_values)}
mlb = MultiLabelBinarizer()
mlb.fit([[label_encoder[label]] for label in label_values])  # Fit the MultiLabelBinarizer
# Load the model with `ignore_mismatched_sizes` set to True
model = T5ForConditionalGeneration.from_pretrained(
    't5-small',
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)
# Step 3: Define the preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(
        examples['sentence'],  # Replace 'text' with the correct column name
        truncation=True,
        padding='max_length',
        max_length=512  # Set the maximum sequence length based on your inputs
    )
    examples['input_ids'] = inputs.input_ids
    examples['attention_mask'] = inputs.attention_mask
    examples['labels'] = mlb.transform([[label_encoder[label]] for label in examples["unfairness_level"]])
    return examples
# Step 4: Preprocess the datasets
encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
encoded_eval_dataset = eval_dataset.map(preprocess_function, batched=True)
# Step 5: Define the compute_metrics function
import numpy as np
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions.flatten(), axis=1)
    
    # Filter out unknown labels (-1)
    unknown_indices = labels != -1
    filtered_labels = labels[unknown_indices]
    filtered_preds = preds[unknown_indices]
    
    return {"accuracy": accuracy_score(filtered_labels, filtered_preds)}
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
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

model_dir = "rd_id_1"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)