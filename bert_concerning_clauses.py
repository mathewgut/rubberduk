############################################################################################################################################################


"""
BEFORE RUNNING THIS CODE ENSURE THAT YOU HAVE INSTALLED THE PYTORCH LIBRARY.
link here: https://pytorch.org/get-started/locally/

Pytorch is a machine learning library used through python. It allows for hand creating Machine Learning models from scratch. It will allow us essentially to create our own model, starting from using a pretrained one (i.e. bert)

PLEASE USE PIP TO INSTALL ALL LIBRARIES THAT ARE USED IN THIS FILE. IF NOT, YOU WILL NOT BE ABLE TO RUN THE CODE OR TRAIN THE MODEL.
"""

"""
Previously, we were using OpenAI's GPT API. Problem being is that training the model costs like 20 dollars an epoch and it is not as customizable.
This version uses Google's BERT-base model, it has around 325M perameters and is good with language processing. This version is free, and can be hosted on a cloud service for significantly
cheaper costs (if ever pushed to production). It was obtained through the huggingface libraries and is already trained, we are just finetuning it (similar to openai api).
The current hitch is data. We switched from idealogies to statements that are concerning and not concerning (it makes more sense given thats the models goal to identify). It is a lot more to read and look at, but allows for
amazing results and with no testing cost (yay). 
"""


"""
2023/05/05 - The model is currently overfitting with the data. We are using F1 scores and losses to determine accuracy. (the lower the loss the more accuate to the data, the higher the f1 the more precise it is.)
It is able to reproduce text very well, and there is no issue with formatting. However, it is having a hard time producing more than one or two lines from the text with the concerning clauses extracton method.
The current theory is that this is due to the way the text is being read in. It is all in one string and isn't formatted. A fix to this would be to use .readlines() so it will read the text one line at a time to the model.
However, it will take longer to process. The model has been trained, the current data lies in the Model Data folder. Overall, we need more data. A lot more data. more concerning and not concerning clauses need to be added.
I have been using NLP's to generate some, however, there are some repeated ones, and there needs to be hundreds of examples for the model to understand nuance. 

The TOS's are used essentially to validate the training. When the model finishes its training loop, it will begin an F1 test, then it will try to produce an output on a random TOS and identify concerning clauses. I've had no luck so far,
but feel free to experiment with it and try to get it working.
"""

"""
This code is very complicated and very hard to understand without context. Give it to a Natural Language Processor such as Bard or ChatGPT to explain how it works and its functionality.
"""

############################################################################################################################################################






import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel, pipeline, XLNetTokenizer, XLNetForSequenceClassification, XLNetConfig
import time
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy
from torch import nn
from torch.optim import AdamW
import re
from data_set import train_data, concerning_list, not_concerning
from clause_audit import audit_concerning_clauses
# this was to limit cuda ram usage, but I am having a lot of issues getting it to set max vram, so its disabled for now
#import os
xlnet_audit = 'xlnet-base-cased'

### Definitions ###




#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5000"
time_current = time.asctime(time.localtime(time.time()))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# model configuration, change the directory to just a name to create a new folder for a new model (eg.config = BertConfig.from_json_file("New Model")) 
config = BertConfig.from_json_file("clause_extract_model\\config.json")
audit_model = XLNetForSequenceClassification.from_pretrained(xlnet_audit, num_labels=2)
# Initialize the model with the configuration
model = BertForSequenceClassification(config)

# Load the pre-trained weights
# Only use if a pre trained model already exists. The weights are the importance of certain attributes it finds in a neural network.
model.load_state_dict(torch.load("clause_extract_model\\pytorch_model.bin"))

twitter_tos = open("twitter_tos.txt", "r", encoding='utf-8')
text2 = twitter_tos.read()
facebook_tos = open("facebook_tos.txt", "r", encoding='utf-8')
text1 = facebook_tos.read()
reddit_tos = open("reddit_tos.txt", "r", encoding='utf-8')
text3 = reddit_tos.read()
youtube_tos = open("youtube_tos.txt", "r", encoding='utf-8')
text4 = youtube_tos.read()
linkedin_tos = open("linkedin_tos.txt", "r", encoding='utf-8')
text5 = linkedin_tos.read()
nytimes_tos = open("nytimes_tos.txt", "r", encoding='utf-8')
text6 = nytimes_tos.read()
openai_tos = open("openai_tos.txt", "r", encoding='utf-8')
text7 = openai_tos.read()
epic_tos = open("epic_tos.txt", "r", encoding='utf-8')
text8 = epic_tos.read()
steam_tos = open("steam_tos.txt", "r", encoding='utf-8')
text9 = steam_tos.read()
#tiktok_tos = open("tiktok_tos.txt", "r", encoding='utf-8')
#text10 = tiktok_tos.read()
playstation_tos = open("playstation_tos.txt", "r", encoding='utf-8')
text11 = playstation_tos.read()
mississauga_tos = open("mississauga_tos.txt", "r", encoding='utf-8')
text12 = mississauga_tos.read()
ea_tos = open("ea_tos.txt", "r", encoding='utf-8')
text13 = ea_tos.read()
betterhelp_tos = open("betterhelp_tos.txt", "r", encoding='utf-8')
text14 = betterhelp_tos.read()

tos_call_list = [text1, text2, text3, text4, text5, text6, text7, text8, text9, #text10,
                  text11, text12, text13, text14]
tos_call_text = random.choice(tos_call_list)
device = torch.device('cpu')  # This is set to just CPU. you can change it by putting GPU instead. The GPU will allow for immensley faster training, but there is no current way to limit VRAM (at least that works)

# encodes the clauses for bert to tokenize and evaluate easier
for clause in concerning_list:
    encoded_clause = tokenizer.encode_plus(clause, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    train_data.append((encoded_clause['input_ids'], encoded_clause['attention_mask'], torch.tensor(1)))

for clause in not_concerning:
    encoded_clause = tokenizer.encode_plus(clause, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    train_data.append((encoded_clause['input_ids'], encoded_clause['attention_mask'], torch.tensor(0)))

# creates a 1 demensional tensor (a series of vectors) to pass in to torch (essentially how we are training the bert model)
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids, attention_mask, label = self.data[index]
        return input_ids.squeeze(), attention_mask.squeeze(), label


def fine_tune_model(train_data, num_epochs=4, batch_size=32, learning_rate=5e-5):
    dataset = CustomDataset(train_data)
    #passes in the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # starts the training
    model.train()#.cuda
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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


# defines that pytorch should train using gpu if avaliable, if not it will use CPU
model.to(device)


#calling the fine tune method
#fine_tune_model(train_data, num_epochs=1, batch_size=32, learning_rate=5e-5)

# testing the model with a seperate data set to test its accuracy, this is f1
def evaluate(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []
    # gradient is a machine learning term, it means what you think it does. It will essentially try the model dry.
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            #MLM or masked learning model is what BERT is. its slightly different than GPT. It covers 15% of the text and tries to guess it. Thats what the mask is here.
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(outputs.logits, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    f1 = f1_score(y_true, y_pred)
    return f1


def extract_concerning_clauses(tos_text, window_size=3):
    # Split tos_text into sentences or lines
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', tos_text)

    model.eval()

    concerning_clauses = []

    for i in range(len(sentences)):
        # Determine the start and end indices of the context window
        start_index = max(0, i - window_size + 1)
        end_index = i + 1

        # Concatenate the lines within the context window
        context = ' '.join(sentences[start_index:end_index])

        encoded_sentence = tokenizer.encode(
            context,
            add_special_tokens=True,
            truncation=True,
            padding='max_length'
        )
        input_ids = torch.tensor(encoded_sentence).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1)

        if predicted_class.item() == 1:
            clause = sentences[i].strip()
            concerning_clauses.append(clause)
    print("concerning clauses")
    print("###################")
    print(concerning_clauses)
    print("###################")
    print("concerning clauses")
    print("###################")
    audit_concerning_clauses(concerning_clauses)
    print("###################")


# Initialize concerning clauses extraction and print found clauses
def concerning_clauses_run():
    concerning_clauses = extract_concerning_clauses(tos_call_text, window_size=3)
    for clause in concerning_clauses:
        print(clause)

# Initalize testing data and tokenize it
eval_data = []
for clause in concerning_list:
    encoded_clause = tokenizer.encode_plus(clause, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    eval_data.append((encoded_clause['input_ids'], encoded_clause['attention_mask'], torch.tensor(1)))

for clause in not_concerning:
    encoded_clause = tokenizer.encode_plus(clause, add_special_tokens=True, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    eval_data.append((encoded_clause['input_ids'], encoded_clause['attention_mask'], torch.tensor(0)))

eval_dataset = CustomDataset(eval_data)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=True)

# Evaluate the model on the evaluation dataset
f1_score = evaluate(model, eval_dataloader)
print("F1 Score:", f1_score)

extract_concerning_clauses(tos_call_text, window_size=3)

#model.save_pretrained("Model Data")

##### XLNET ######
"""

def evaluate_audit(audit_model, dataloader):
    audit_model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = audit_model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(outputs.logits, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    f1 = f1_score(y_true, y_pred)
    return f1


f1_score = evaluate_audit(audit_model, eval_dataloader)
print("F1 Score:", f1_score)

# Function to audit the concerning clauses




#results = audit_concerning_clauses(concerning_clauses)
# Print the audit results
for clause, predicted_label in results:
    if predicted_label == 1:
        print(f"Concerning clause: {clause}")
    else:
        print(f"Not concerning clause: {clause}")


"""

# Close the file handles
for tos_file in [twitter_tos, facebook_tos, reddit_tos, youtube_tos, linkedin_tos, nytimes_tos, openai_tos, epic_tos, steam_tos, #tiktok_tos
                 ]:
    tos_file.close()

