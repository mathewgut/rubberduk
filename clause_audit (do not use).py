from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetConfig
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy
from torch import nn
from torch.optim import AdamW
import torch
from bert_concerning_clauses import CustomDataset, train_data

xlnet_audit = 'xlnet-base-cased'
tokenizer_audit = XLNetTokenizer.from_pretrained(xlnet_audit, truncation=True, max_length=512)
audit_model = XLNetForSequenceClassification.from_pretrained(xlnet_audit, num_labels=2)
from bert_concerning_clauses import concerning_clauses, not_concerning, extract_concerning_clauses, tos_call_text, concerning_clauses_run


device = torch.device('cpu')


audit_model.to(device)
def fine_tune_audit(train_data, num_epochs=4, batch_size=32, learning_rate=5e-5):
    dataset = CustomDataset(train_data)
    #passes in the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # starts the training
    audit_model.train()#.cuda
    optimizer = torch.optim.AdamW(audit_model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    # training loop with number of loops
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0  # Track total loss (inaccuracy) for the epoch
        # current loop
        for step, batch in enumerate(dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            # gives labels to the model for identifying
            labels = batch[2].to(device)

            optimizer.zero_grad()
            # compared against the masked models to compare accuracy of guessed words
            outputs = audit_model(input_ids, attention_mask=attention_mask, labels=labels)
            # inaccuracy of output
            loss = outputs.loss
            # backward propagates. Essentially once the model finishes passing through the neural network and gets its answer, 
            # it will go backwards and adjust the weights (the importance of a certain area of the neural network to achieving an accurate answer) 
            # to hopefully make a more accurate prediction next time
            # this is the backbone of how machine learning works with Nerual Networks
            loss.backward()
            # optimizes the step with the new weights
            optimizer.step()

            total_loss += loss.item()
            # no idea what this part is
            if (step + 1) % 10 == 0:
                average_loss = total_loss / (step + 1)
                print(f"Step {step + 1}, Average Loss: {average_loss:.4f}")
        # calculates the toal incaccuracy and prints the value
        average_loss = total_loss / (step + 1)
        print(f"Epoch {epoch + 1} Average Loss: {average_loss:.4f}")

<<<<<<< HEAD:clause_audit (do not use).py
fine_tune_audit(train_data, num_epochs=5, batch_size=32, learning_rate=5e-5)
=======
#fine_tune_audit(train_data, num_epochs=1, batch_size=32, learning_rate=5e-5)
>>>>>>> 5ca2c0a1cf5887f50bf4df9382935dd5339d2b79:clause_audit
audit_model.save_pretrained("audit_model_xlnet")
config = XLNetConfig.from_json_file("audit_model_xlnet\\config.json")
audit_model = XLNetForSequenceClassification(config)
audit_model.load_state_dict(torch.load("audit_model_xlnet\\pytorch_model.bin"))

def audit_concerning_clauses(concerning_clauses):
    context = "data privacy, data selling, security, cross site tracking/monitoring, data rights"
    audit_model.eval()
    audit_results = []

    for clause in concerning_clauses:
        # Combine the clause and context
        text = f"{context} {clause}"

        # Tokenize the text
        inputs = tokenizer_audit.encode_plus(text, add_special_tokens=True, padding='longest', return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Perform inference
        with torch.no_grad():
            outputs = audit_model(input_ids, attention_mask=attention_mask)

        # Get the predicted label
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        audit_results.append((clause, predicted_label))

    print(audit_results)
<<<<<<< HEAD:clause_audit (do not use).py
    return audit_results
=======
    return audit_results


### Ask user if they wish to save model###
print("Would you like to save the model?")
save_model_input = input("[y/n]")

if save_model_input == "y":
    audit_model.save_pretrained("audit_model_xlnet")
elif save_model_input == "n":
    pass
else:
    save_model_input

concerning_clauses_run()
>>>>>>> 5ca2c0a1cf5887f50bf4df9382935dd5339d2b79:clause_audit
