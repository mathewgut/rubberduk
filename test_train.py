from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("joelito/online_terms_of_service")

# Load the T5 model and tokenizer
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    # Tokenize the inputs and targets
    inputs = [f"translate English to English: {text}" for text in examples["text"]]
    targets = examples["text"]
    
    # Tokenize inputs and targets
    tokenized_inputs = tokenizer.batch_encode_plus(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized_targets = tokenizer.batch_encode_plus(
        targets,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Set the appropriate input and target fields
    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    tokenized_inputs["decoder_input_ids"] = tokenized_targets["input_ids"]
    
    return tokenized_inputs

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./t5-small-terms-of-service",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_total_limit=1,
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir="./t5-small-terms-of-service/logs",
    overwrite_output_dir=True,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)


# Train the model
trainer.train()

# Save the model with a custom name
model_dir = "rd_id_1"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
