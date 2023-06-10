from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import random
import nltk
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
#from clause_expain import analyze
#import numpy as np
#import networkx as nx
import os
from flask import Flask

cloud_app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

"""

HOW THIS CODE WORKS:

This code utilizes three key concepts.

1. Zero Shot Classification:

This is where the model uses its generalized knowledge from its training data to classify inputs into labels (that we define) without any finetuning on the specific tasks/labels.

2. Multi Model Fusion:

To achieve a more balanced (and hopefully more accurate) output, we are combining two different models abilities and outputs. Currently these models are: Bart (trained on MLNI) and de-berta-base (trained on SNLI and MNLI). By processing
the same inputs into both models, and checking to see if a model is more than 70% confident an input is concerning we put each models confidence score along with the input we then compare results between the two lists.
If both models think the same input is concerning (with a confidence of 70% or more), we add that text to the mutual likely than average the confidence score between the two models. For this, we are using Early Model Fusion.

3. Summarization

We use a third model (trained for summarization) to summarize the list of concerning clauses/terms in the document so the user has a breakdown of what is found to be concerning. Hopefully, in time, we will add features that break down
why a certain point is concerning in more detail. The average confidence score can be used as an indicator for how concerning the TOS/EULA/Privacy Policy may be overall.

How does input and batch size work and what happens?:

When a text is input (i.e a tos/eula/privacy policy) the document is read line by line (so if there is only one line for the entire document or formatting is bad, you will get a bad result), this is because we want to control the batch size
or number of lines the model is reading at once to allow for more precise classifcations of certain passages of text. The model (currently) only classifies the specified batches at a time without context from previous or next batches.
This can cause errors (especially with short texts) where the model will think something is extremley concerning, or not concerning, as it doesn't know the context of which the text is used in. The batch size is very important for accuracy
as it is the only current control of context for zero shot classification.

How do I use the concerning clause identifier?:

If you wish to use just one model at a time you can just use clause_identifier. It holds a bunch of lists with information such as non concerning, and concerning texts and their scores.
    To use this, pass in the following values: 
    document: this is the text you wish to read, make sure it is readable and formatted not horribly
    classifier: this is the model you wish to use for zero shot classification (ideally from transformers pipeline for better performance)
    candidate_labels: these are the labels you wish to use for classification. You can have as many labels as you want, but, it will only store values for labels in postion 0 and 1 (positive and negative/concerning and not concerning).
    speech: this one is pretty stupid, but essentially it is what you want to say once process finishes. I put quacks because rubberduk... get it? Ha... funny funny man.
    batch: this is the amount of lines you wish for it to read in at once. The amount of lines you give it will determine the quality of the classification (along with labels). The more lines, the more context, but less precise and vice versa.
    c_temp: The threshold of confidence the model has to reach for label pos 0 to be added to the list.
    nc_temp: The threshold of confidence the model has to reach for label pos 1 to be added to the list.

If you want to use multi model fusion, use fusion_model_general, this will give (typically) more accurate results than just using one model. This holds even more information including clean text of all the concerning clauses,
averaged scores, individual scores, the current lines in the batch list, the clean results of the scores per batch, and lots of other stuff.
    To use this, pass in the following values:
    size: This is batch size, the amount of lines you wish for it to read in at a time. The amount of lines you give it will determine the quality of the classification (along with labels). The more lines, the more context, but less precise and vice versa.
    labels: These are the labels/classes you wish to give the model to match with given text. The wording used is crucial to task accuracy, so make sure you use clear text that reflects your task (concerning clause, not concerning clause)
    output_file: This value should look something like this: "file_privacy = open("output_privacy.txt", "w")", in this instance, file_privacy is what we would pass in to fusion_model_general. You need to give it write or append access to your designated file, or it won't write anything (duh). 
    concern_temp: The threshold of confidence the model has to reach for label pos 0 to be added to the list.
    no_concern_temp: The threshold of confidence the model has to reach for label pos 1 to be added to the list.

You're ready to use the code! Please don't break anything, my insurance on this place just lapsed so I'd be screwed.

yours truly,

- fart master extreme

"""

## TODO: Enhance multi model fusion accuracy by adding more models to lessen the weight of each models classification confidence scores (averaging for more accuracy)
## TODO: Gather more TOS's for evaluation (find ones that are actually concerning *cough* *cough* facebook, tiktok)
## TODO: Restructure fusion function to add models likely concerning text to individual lists, than compare and contrast all models at once (will allow for increased scale)
## TODO: Optimize code and rid of bloat and random functions (get rid of zoo_wee_mama :C )
## TODO: attatch to frontend (js) to allow to read in texts found on the web
## TODO: Fix summarizer function so the almalgomated summary is not a bajilion words long
## TODO: Use BaptisteDoyen/camembert-base-xnli in multi model fusion, accuracy tests well
## TODO: Make fusion function only return clauses that meet an average confidence criteria (i.e. 70%, 80%, etc)
## TODO: Add to cloud hosting (AWS, Google Cloud, Azure) for increased performance, and for utilization with a chrome extension
## TODO: Overhaul URL Scraping function (have it actually work) and create a formatting function for scaped files to rid of special characters, spacing issues, and other junk
## TODO: Add a language input to change the models for reading clauses in different languages (french, spanish, italian, portugese, etc)




# these are the files used for testing
twitter_tos = open("twitter_tos.txt", "r", encoding='utf-8')
text2 = twitter_tos.readlines()
facebook_tos = open("facebook_tos.txt", "r", encoding='utf-8')
text1 = facebook_tos.readlines()
reddit_tos = open("reddit_tos.txt", "r", encoding='utf-8')
text3 = reddit_tos.readlines()
youtube_tos = open("youtube_tos.txt", "r", encoding='utf-8')
text4 = youtube_tos.readlines()
linkedin_tos = open("linkedin_tos.txt", "r", encoding='utf-8')
text5 = linkedin_tos.readlines()
nytimes_tos = open("nytimes_tos.txt", "r", encoding='utf-8')
text6 = nytimes_tos.readlines()
openai_tos = open("openai_tos.txt", "r", encoding='utf-8')
text7 = openai_tos.readlines()
epic_tos = open("epic_tos.txt", "r", encoding='utf-8')
text8 = epic_tos.readlines()
steam_tos = open("steam_tos.txt", "r", encoding='utf-8')
text9 = steam_tos.readlines()
playstation_tos = open("playstation_tos.txt", "r", encoding='utf-8')
text11 = playstation_tos.readlines()
mississauga_tos = open("mississauga_tos.txt", "r", encoding='utf-8')
text12 = mississauga_tos.readlines()
ea_tos = open("ea_tos.txt", "r", encoding='utf-8')
text13 = ea_tos.readlines()
betterhelp_tos = open("betterhelp_tos.txt", "r", encoding='utf-8')
text14 = betterhelp_tos.readlines()
tiktok_tos = open("tiktok_tos.txt", "r", encoding='utf-8')
text15 = tiktok_tos.readlines()
netflix_tos = open("netflix_tos.txt", "r", encoding='utf-8')
text16 = netflix_tos.readlines()


tos_call_list = [text1, text2, text3, text4, text5, text6, text7, text8, text9,
                  text11, text12, text13, text14, text15, text16]
# assigning a random position based on a range from 0 to the length of the list
tos_call_text = tos_call_list[random.choice(range(0, len(tos_call_list)))]


# what the model says after finishing a classification in its entirety
quacks = "Thanks for playing, quack"
summary_text_list = []
@cloud_app.route("/summarize")
def smart_summarize(text):
    # Initialize the BART summarization pipeline
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

    # Check the length of the input text
    total_length = len(text)

    if total_length <= 1024:
        # Input text is within the character limit, summarize it directly
        summary = summarizer(text, max_length=500)[0]['summary_text']
        return summary

    else:
        # Calculate the number of sections needed
        section_length = 512  # Adjust the section length as desired
        overlap = 128  # Adjust the overlap length as desired

        # Create overlapping sections using a sliding window
        sections = []
        start = 0
        while start < total_length:
            end = min(start + section_length, total_length)
            sections.append(text[start:end])
            start += section_length - overlap

        # Summarize each section
        summaries = [summarizer(section)[0]['summary_text'] for section in sections]

        # Concatenate the summaries
        summary = ' '.join(summaries)

        return summary


# simple function for finding the mean of a list of ints/floats
def mean(x):
    total = sum(x) / len(x)
    return total
# the engine?
@cloud_app.route("/identifier")
def clause_identifier(document, classifier, candidate_labels, speech, batch, c_temp, nc_temp):
    # currently not used value for extreme confidence scoring
    v_avg_concerning = []
    # stores concerning confidence values for averaging
    avg_concerning = []
    # stores non concerning confidence values for averaging
    avg_not_concerning = []
    # blanket lists for other labels (if used)
    avg_ambiguous = []
    avg_legal_concern = []
    # a fail safe for the while loop if the end of a document is reached
    trigger = True
    # the results for both concerning and non concerning confidence values with the batch text used as a key
    results =	{}
    # the list of text used for identification in current loop
    batch_list = []
    # texts and their confidence values that have passed the c_temp threshold
    likely_concerning = []
    # texts and their confidence values that have passed the nc_temp threshold
    likely_not_concerning = []


    while trigger == True:
            # for determining how many loops have passed
            counter = 0
            # for each line in provided text
            for y in document:
                    # if the amount of loops is not equal to the specified batch size
                    if counter < batch:
                        counter += 1
                        # add current line to batch list
                        batch_list.append(y)
                        continue
                    # if number of loops == the specified batch size
                    elif counter >= batch:
                        # make all lines in the batch_list into one big string seperated by a space
                        y = " ".join(batch_list)
                        # clear batch list
                        batch_list = []
                        # clear loop counter
                        counter = 0
                    print(y)
                    # unnecessary, but funny
                    zoo_wee_mama = classifier(y, candidate_labels, multi_label=False)
                    # add current line as a key to the results dict and set the confidence scores as the value
                    results.update({y: zoo_wee_mama['scores']})
                    # add the concerning confidence score to avg_concerning for averaging
                    avg_concerning.append(zoo_wee_mama["scores"][0])
                    # if score in pos 0 (concerning) passes the set threshold, add to likely concerning
                    if zoo_wee_mama["scores"][0] > c_temp:
                        likely_concerning.append([y, zoo_wee_mama["scores"][0]])
                    # add the non concerning confidence score to avg_not_concerning for averaging
                    avg_not_concerning.append(zoo_wee_mama["scores"][1])
                    # if score in pos 1 (not concerning) passes the set threshold, add to likely not concerning
                    if zoo_wee_mama["scores"][1] > nc_temp:
                        likely_not_concerning.append([y, zoo_wee_mama["scores"][1]])
                    # print the current averages for concerning and not concerning
                    print("avg concerning: ", mean(avg_concerning))
                    print("avg not concerning: ", mean(avg_not_concerning))  
                    
                #print(results)
            trigger = False
            break
    # print final outcome statements of classification on text
    print("########################")
    print(results)
    print("Totals:")
    #print("total v avg concerning: ", mean(v_avg_concerning))
    print("total avg concerning: ", mean(avg_concerning))
    print("total avg not concerning: ", mean(avg_not_concerning))
    #print("total avg ambiguous: ", mean(avg_ambiguous))
    print(likely_concerning)
    print(likely_not_concerning)
    print(speech)
    return likely_concerning


@cloud_app.route("/fusion")
# for fusion of two models
def fusion_model_general(size,labels,output_file, concern_temp, no_concern_temp):
    # the path for the first model
    model = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    fart = True
    # model 1 likely concerning batches and their confidence scores 
    bart_likely = []
    # model 2 likely concerning batches and their confidence scores 
    valhalla_likely = []
    # shared concerning batches and their scores
    shared_likely = []
    # a list for all scores to be added for averaging
    shared_likely_math = []
    # batches with the average concerning confidence score between the two models
    shared_likely_mean = []
    # a clean list containing only batch text for summarization
    shared_likely_text = []
    
    # to ensure only two labels are used
    try:
        assert(len(labels) == 2)
    except AssertionError:
        print("More than 2 labels found. Please include only one positive and one negative label.")
        return 0
    
    # add the results of the first model
    bart_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size, concern_temp, no_concern_temp))

    print("#####SECOND MODEL#####")
    # change model to second model
    model = pipeline("zero-shot-classification",
            model="cross-encoder/nli-deberta-base")
    # add the results of the second model
    valhalla_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size, concern_temp, no_concern_temp))
    
    print("######b_likely#######")
    # print likely results of first model
    print(bart_likely)
    print("######v_likely#######")
    # print likely results of second model
    print(valhalla_likely)
    
    
    # since everything is in a list of a list of a list of a list, it has to be accessed this way. sorry :)
    for x in bart_likely:
        for x2 in x:
            for y in valhalla_likely:
                for y2 in y:
                    # if the current batch text of model one matches the current batch text of model 2
                    if x2[0] == y2[0]:
                        # add the batch text along with both confidence scores
                        shared_likely.append([x2[0], x2[1], y2[1]])
                        # add both confidence scores to shared_likely_math for averaging
                        shared_likely_math.append(x2[1])
                        shared_likely_math.append(y2[1])
                        # create the average confidence score between the two models per batch
                        avg_shared = sum(shared_likely_math) / len(shared_likely_math)
                        # add the average confidence score along with the batch text to shared_likely_mean
                        shared_likely_mean.append((x2[0], avg_shared))
                        # add just the batch text to shared_likely_text
                        shared_likely_text.append(x2[0])
                        # print the number of shared likely concerning batches
                        print("shared =", len(shared_likely))
                        print(shared_likely_mean)
                    # if the text doesn't match print no match and move on to next comparison
                    else:
                        print("no match")
                        continue
    # final outcome print statements
    print("shared =", len(shared_likely))
    print(shared_likely)
    print(shared_likely_mean)
    
    # using try to create an average of the completed lists of concerning confidence scores. 
    # this is because if there is a model with no concerning batches, than the list will be empty so the math will fail and break the code.
    # having an except allows it to be handled and gives the ability for the model to also find no concerning statements within a text
    try:
        avg_shared = sum(shared_likely_math)/len(shared_likely_math)
        print("Shared avg: ", avg_shared)
    except:
        print("No likely concerns found")

    # in case of error to writing to file 
    try:
        for x in shared_likely:
            output_file.writelines(str(x) + "\n")
    except Exception as e:
            print("Error occurred while writing to file:", str(e))
    return shared_likely_text

@cloud_app.route("/analysis")
def analysis_generate(context, label):
    #analysis_tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
    #analysis_model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
    analysis_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    analysis_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    #analysis_model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
    #instruction = f'Instruction: The following is an excerpt text from a TOS/EULA/Privacy Policy. It has been flagged by an LLM as potentially concerning for a user of the platform/service. Please explain which parts a user should be concerned about and why.:'
    instruction = f'Instruction: The following excerpt has been labelled as {label}, please explain which parts of this excerpt apply to this label and why the user would want to be aware of it. Excerpt: '
    query = f"{instruction} {context}"
    input_ids = analysis_tokenizer.encode(query, return_tensors="pt")
    outputs = analysis_model.generate(input_ids, max_length=1024, min_length=20, top_p=0.9, do_sample=True)
    output = analysis_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output)
    return output

    

### calling the functions and defining their needed values


### General Model Fusion
file_general = open("output_general.txt", "w")
classes_general = ['Concerning Clause for User','Non-Concerning Clause for User']
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_general[0]

likely_found = fusion_model_general(5,classes_general, file_general, 0.7,0.6)
text_for_summary = ",".join(likely_found)

summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    #print(summarizer(text_for_summary, max_length=2048, min_length=30, do_sample=False))

explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_general[0])
    explained_text.append(analyze_text)
file_general.write(str(explained_text))
file_general.close()
"""
try:
    summary_text = smart_summarize(text_for_summary)
    print(summary_text)
    summary_text_list.append(summary_text)
#summarize_concerns(text_for_summary)
except:
    print("Summary fail")
"""
print("##############################")


### Data Model Fusion
classes_data = ['Potential Data Use Concern','No Data Use Concern']
file_data = open("output_data.txt", "w")
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_data[0]

likely_found = fusion_model_general(5, classes_data, file_data,0.8,0.5)
text_for_summary = ",".join(likely_found)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_data[0])
    explained_text.append(analyze_text)
file_data.write(str(explained_text))
file_data.close()
"""
try:
    summary_text = smart_summarize(text_for_summary) 
    print(summary_text)
    summary_text_list.append(summary_text)
#summarize_concerns(text_for_summary)
except:
    print("Summary fail")
"""
print("##############################")


### Security Model Fusion
classes_security = ['Potential Security Concern for User','No Security Concern for User']
file_security = open("output_security.txt", "w")
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_security[0]

likely_found = fusion_model_general(5,classes_security,file_security,0.75,0.5)
text_for_summary = ",".join(likely_found)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_security[0])
    explained_text.append(analyze_text)
file_security.write(str(explained_text))
file_security.close()
"""
try:
    summary_text = smart_summarize(text_for_summary)
    print(summary_text)
    summary_text_list.append(summary_text)
#summarize_concerns(text_for_summary)
except:
    print("Summary fail")
"""
print("##############################")


### Privacy Model Fusion
classes_privacy = ['Potential Privacy Concern','No Privacy Concern ']
file_privacy = open("output_privacy.txt", "w")
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_privacy[0]

likely_found = fusion_model_general(5, classes_privacy, file_privacy,0.8,0.5)
text_for_summary = ",".join(likely_found)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_privacy[0])
    explained_text.append(analyze_text)
file_privacy.write(str(explained_text))
file_privacy.close()
"""
try:
    summary_text = smart_summarize(text_for_summary)
    print(summary_text)
    summary_text_list.append(summary_text)
    #summarize_concerns(text_for_summary)
except:
    print("Summary fail")
"""
print("##############################")

### Legal Model Fusion
classes_legal = ['Potential Legal Concern','No Legal Concern ']
file_legal = open("output_legal.txt", "w")
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_legal[0]

likely_found = fusion_model_general(5, classes_legal, file_legal,0.8,0.5)
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_legal[0])
    explained_text.append(analyze_text)
file_legal.write(str(explained_text))
file_legal.close()

text_for_summary = ",".join(likely_found)
"""
try:
    summary_text = smart_summarize(text_for_summary)
    print(summary_text)
    summary_text_list.append(summary_text)
#summarize_concerns(text_for_summary)
except:
    print("Summary fail")
"""
print("##############################")


summary_output = open("output_summary.txt", "w")
print(summary_text_list)
char_count = 0
for x in summary_text_list:
    summary_output.write(str(x))
    for y in x:
        char_count += 1
print(char_count)
#summary_output.write(str("Token Count: ", str(char_count)))
if __name__ == "__clause_identifier_0.2__":
    cloud_app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))