from transformers import pipeline
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import networkx as nx
nltk.download('punkt')
nltk.download('stopwords')
"""

HOW THIS MODEL WORKS:

This model utilizes three key concepts.

1. Zero Shot Classification:

This is where the model uses its generalized knowledge from its training data to classify inputs into labels (that we define) without any finetuning on the specific tasks/labels.

2. Multi Model Fusion:

To achieve a more balanced (and hopefully more accurate) output, we are combining two different models abilities and outputs. Currently these models are: Bart (trained on MLNI) and de-berta-base (trained on SNLI and MNLI). By processing
the same inputs into both models, and checking to see if a model is more than 70% confident an input is concerning we put each models confidence score along with the input we then compare results between the two lists.
If both models think the same input is concerning (with a confidence of 70% or more), we add that text to the mutual likely than average the confidence score between the two models. For this, we are using Early Model Fusion.

3. Summarization

We use a third model (trained for summarization) to summarize the list of concerning clauses/terms in the document so the user has a breakdown of what is found to be concerning. Hopefully, in time, we will add features that break down
why a certain point is concerning in more detail. The average confidence score can be used as an indicator for how concerning the TOS/EULA/Privacy Policy may be overall.


"""



## This code is horribly non-optimized because I used zero chatgpt lmao

#from data_set import concerning_list, not_concerning
model = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

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
#tiktok_tos = open("tiktok_tos.txt", "r", encoding='utf-8')
#text10 = tiktok_tos.read()
playstation_tos = open("playstation_tos.txt", "r", encoding='utf-8')
text11 = playstation_tos.readlines()
mississauga_tos = open("mississauga_tos.txt", "r", encoding='utf-8')
text12 = mississauga_tos.readlines()
ea_tos = open("ea_tos.txt", "r", encoding='utf-8')
text13 = ea_tos.readlines()
betterhelp_tos = open("betterhelp_tos.txt", "r", encoding='utf-8')
text14 = betterhelp_tos.readlines()


tos_call_list = [text1, text2, text3, text4, text5, text6, text7, text8, text9, #text10,
                  text11, text12, text13, text14]
tos_call_text = tos_call_list[random.choice(range(0, len(tos_call_list)))]
#tos_call_text = random.choice(tos_call_list)



quacks = "Thanks for playing, quack"

def sentence_similarity(sent1, sent2):
    words1 = word_tokenize(sent1)
    words2 = word_tokenize(sent2)
    words1 = set(words1)
    words2 = set(words2)
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    similarity = len(intersection) / len(union)
    return similarity

def summarize_concerns(text, summary_length=5):
    # Preprocess the text
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    sentences = sent_tokenize(text)
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [porter.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
        preprocessed_sentences.append(' '.join(words))

    # Create a similarity matrix between sentences
    similarity_matrix = np.zeros((len(preprocessed_sentences), len(preprocessed_sentences)))
    for i in range(len(preprocessed_sentences)):
        for j in range(len(preprocessed_sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(preprocessed_sentences[i], preprocessed_sentences[j])

    # Apply PageRank algorithm to get sentence scores
    scores = nx.pagerank(nx.from_numpy_array(similarity_matrix))

    # Sort the sentences by their scores in descending order
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(preprocessed_sentences)), reverse=True)

    # Select the top sentences to form the summary
    top_sentences = ranked_sentences[:summary_length]

    # Join the top sentences to form the summary
    summary = ' '.join(sentence for _, sentence in top_sentences)

    return summary


def mean(x):
    total = sum(x) / len(x)
    return total
# the engine?
def clause_identifier(document, classifier, candidate_labels, speech, batch):
    v_avg_concerning = []
    avg_concerning = []
    avg_not_concerning = []
    avg_ambiguous = []
    avg_legal_concern = []
    fart = True
    results =	{}
    batch_list = []
    likely_concerning = []
    likely_not_concerning = []


    while fart == True:
            counter = 0
            for y in document:
                    if counter < batch:
                        counter += 1
                        batch_list.append(y)
                        continue
                    elif counter >= batch:
                        y = " ".join(batch_list)
                        batch_list = []
                        counter = 0
                    print(y)
                    #unnecessary, but funny
                    zoo_wee_mama = classifier(y, candidate_labels, multi_label=False)
                    #zoo_we_mama_scores = str(zoo_wee_mama[1]) + str(zoo_wee_mama[2])
                    results.update({y: zoo_wee_mama['scores']})
                    #v_avg_concerning.append(zoo_wee_mama["scores"][0])
                    avg_concerning.append(zoo_wee_mama["scores"][0])
                    if zoo_wee_mama["scores"][0] > 0.7:
                        likely_concerning.append([y, zoo_wee_mama["scores"][0]])
                    avg_not_concerning.append(zoo_wee_mama["scores"][1])
                    if zoo_wee_mama["scores"][1] > 0.60:
                        likely_not_concerning.append([y, zoo_wee_mama["scores"][1]])
                    #avg_ambiguous.append(zoo_wee_mama["scores"][2])
                    #avg_legal_concern.append(zoo_wee_mama["scores"][4])
                    #print("avg concerning: ", mean(v_avg_concerning))
                    print("avg concerning: ", mean(avg_concerning))
                    print("avg not concerning: ", mean(avg_not_concerning))  
                    #print("avg ambiguous: ", mean(avg_ambiguous))
                    #print("avg legal concern:: ", mean(avg_legal_concern))   
                #print(results)
            fart = False
            break
    
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




def fusion_model_general(size):
    model = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    fart = True
    bart_likely = []
    valhalla_likely = []
    shared_likely = []
    shared_likely_math = []
    shared_likely_mean = []
    shared_likely_text = []
    

    labels = ['Concerning Clause for User','Non-Concerning Clause for User']
    #labels = ['Terms of Use Alert','Non-Alerting Term']
    bart_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size))
   
    

    print("#####SECOND MODEL#####")
    """
    model = pipeline("zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1")
            """
    model = pipeline("zero-shot-classification",
            model="cross-encoder/nli-deberta-base")
    valhalla_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size))
    
    print("######b_likely#######")
    print(bart_likely)
    print("######v_likely#######")
    print(valhalla_likely)
    
    
    """
    labels = ['Unethical Use of Data','Ethical Use of Data']
    clause_identifier(concerning_list, model, labels, quacks)
    """
    
    """
    for x in bart_likely:
        for y in valhalla_likely:
            if x[0] == y[0]:
                shared_likely.append([x[0],x[1],y[1]])
                shared_likely_math.append(x[1])
                shared_likely_math.append(y[1])
                avg_shared = sum(shared_likely_math)/len(shared_likely_math)
                shared_likely_mean.append(x[0],avg_shared)
                print("shared = ", len(shared_likely))
                print(shared_likely_mean)
            else:
                pass
    print(shared_likely)
    """
    for x in bart_likely:
        for x2 in x:
            for y in valhalla_likely:
                for y2 in y:
                    if x2[0] == y2[0]:
                        shared_likely.append([x2[0], x2[1], y2[1]])
                        shared_likely_math.append(x2[1])
                        shared_likely_math.append(y2[1])
                        avg_shared = sum(shared_likely_math) / len(shared_likely_math)
                        shared_likely_mean.append((x2[0], avg_shared))
                        shared_likely_text.append(x2[0])
                        print("shared =", len(shared_likely))
                        print(shared_likely_mean)
                    else:
                        print("no match")
                        continue

    print(shared_likely)
    print(shared_likely_mean)

    if len(valhalla_likely) == 0:
         print("v found no concerning clauses over 65 percent confidence")
         return bart_likely
    elif len(bart_likely) == 0:
         print("b found no concerning clauses over 65 percent confidence")
         return valhalla_likely
    elif len(valhalla_likely) == 0 and len(bart_likely) == 0:
        print("b and v found no concerning clauses over 65 percent confidence")
        return 0
    else:
        avg_shared = sum(shared_likely_math)/len(shared_likely_math)
        print("Shared avg: ", avg_shared)
        output_file = open("output_general.txt", "w")
        for x in shared_likely_mean:
            output_file.write(str(x))
        return shared_likely_text
    
def fusion_model_data(size):
    model = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    fart = True
    bart_likely = []
    valhalla_likely = []
    shared_likely = []
    shared_likely_math = []
    shared_likely_mean = []
    shared_likely_text = []
    

    labels = ['Data Concern for User','No Data Concern for User']
    #labels = ['Terms of Use Alert','Non-Alerting Term']
    bart_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size))
   
    

    print("#####SECOND MODEL#####")
    """
    model = pipeline("zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1")
            """
    model = pipeline("zero-shot-classification",
            model="cross-encoder/nli-deberta-base")
    valhalla_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size))
    
    print("######b_likely#######")
    print(bart_likely)
    print("######v_likely#######")
    print(valhalla_likely)
    

    for x in bart_likely:
        for x2 in x:
            for y in valhalla_likely:
                for y2 in y:
                    if x2[0] == y2[0]:
                        shared_likely.append([x2[0], x2[1], y2[1]])
                        shared_likely_math.append(x2[1])
                        shared_likely_math.append(y2[1])
                        avg_shared = sum(shared_likely_math) / len(shared_likely_math)
                        shared_likely_mean.append((x2[0], avg_shared))
                        shared_likely_text.append(x2[0])
                        print("shared =", len(shared_likely))
                        print(shared_likely_mean)
                    else:
                        print("no match")
                        continue

    print(shared_likely)
    print(shared_likely_mean)

    if len(valhalla_likely) == 0:
         print("v found no concerning clauses over 65 percent confidence")
         return bart_likely
    elif len(bart_likely) == 0:
         print("b found no concerning clauses over 65 percent confidence")
         return valhalla_likely
    elif len(valhalla_likely) == 0 and len(bart_likely) == 0:
        print("b and v found no concerning clauses over 65 percent confidence")
        return 0
    else:
        avg_shared = sum(shared_likely_math)/len(shared_likely_math)
        print("Shared avg: ", avg_shared)

        output_file = open("output_data.txt", "w")
        for x in shared_likely_mean:
            output_file.write(str(x))
        return shared_likely_text
    
def fusion_model_security(size):
    model = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    fart = True
    bart_likely = []
    valhalla_likely = []
    shared_likely = []
    shared_likely_math = []
    shared_likely_mean = []
    shared_likely_text = []
    

    labels = ['Security Concern for User','No Security Concern for User']
    #labels = ['Terms of Use Alert','Non-Alerting Term']
    bart_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size))
   
    

    print("#####SECOND MODEL#####")
    """
    model = pipeline("zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1")
            """
    model = pipeline("zero-shot-classification",
            model="cross-encoder/nli-deberta-base")
    valhalla_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size))
    
    print("######b_likely#######")
    print(bart_likely)
    print("######v_likely#######")
    print(valhalla_likely)
    

    for x in bart_likely:
        for x2 in x:
            for y in valhalla_likely:
                for y2 in y:
                    if x2[0] == y2[0]:
                        shared_likely.append([x2[0], x2[1], y2[1]])
                        shared_likely_math.append(x2[1])
                        shared_likely_math.append(y2[1])
                        avg_shared = sum(shared_likely_math) / len(shared_likely_math)
                        shared_likely_mean.append((x2[0], avg_shared))
                        shared_likely_text.append(x2[0])
                        print("shared =", len(shared_likely))
                        print(shared_likely_mean)
                    else:
                        print("no match")
                        continue

    print(shared_likely)
    print(shared_likely_mean)

    if len(valhalla_likely) == 0:
         print("v found no concerning clauses over 65 percent confidence")
         return bart_likely
    elif len(bart_likely) == 0:
         print("b found no concerning clauses over 65 percent confidence")
         return valhalla_likely
    elif len(valhalla_likely) == 0 and len(bart_likely) == 0:
        print("b and v found no concerning clauses over 65 percent confidence")
        return 0
    else:
        avg_shared = sum(shared_likely_math)/len(shared_likely_math)
        print("Shared avg: ", avg_shared)
        
        output_file = open("output_security.txt", "w")
        for x in shared_likely_mean:
            output_file.write(str(x))
        
        return shared_likely_text


def fusion_model_privacy(size):
    model = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    fart = True
    bart_likely = []
    valhalla_likely = []
    shared_likely = []
    shared_likely_math = []
    shared_likely_mean = []
    shared_likely_text = []
    

    labels = ['Privacy Concern for User','No Privacy Concern for User']
    #labels = ['Terms of Use Alert','Non-Alerting Term']
    bart_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size))
   
    

    print("#####SECOND MODEL#####")
    """
    model = pipeline("zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1")
            """
    model = pipeline("zero-shot-classification",
            model="cross-encoder/nli-deberta-base")
    valhalla_likely.append(clause_identifier(tos_call_text, model, labels, quacks, size))
    
    print("######b_likely#######")
    print(bart_likely)
    print("######v_likely#######")
    print(valhalla_likely)
    

    for x in bart_likely:
        for x2 in x:
            for y in valhalla_likely:
                for y2 in y:
                    if x2[0] == y2[0]:
                        shared_likely.append([x2[0], x2[1], y2[1]])
                        shared_likely_math.append(x2[1])
                        shared_likely_math.append(y2[1])
                        avg_shared = sum(shared_likely_math) / len(shared_likely_math)
                        shared_likely_mean.append((x2[0], avg_shared))
                        shared_likely_text.append(x2[0])
                        print("shared =", len(shared_likely))
                        print(shared_likely_mean)
                    else:
                        print("no match")
                        continue

    print(shared_likely)
    print(shared_likely_mean)

    if len(valhalla_likely) == 0:
         print("v found no concerning clauses over 65 percent confidence")
         return bart_likely
    elif len(bart_likely) == 0:
         print("b found no concerning clauses over 65 percent confidence")
         return valhalla_likely
    elif len(valhalla_likely) == 0 and len(bart_likely) == 0:
        print("b and v found no concerning clauses over 65 percent confidence")
        return 0
    else:
        avg_shared = sum(shared_likely_math)/len(shared_likely_math)
        print("Shared avg: ", avg_shared)
        output_file = open("output_privacy.txt", "w")
        for x in likely_found:
            output_file.write(str(x))


### General Model Fusion
#try:
likely_found = fusion_model_general(5)
text_for_summary = ",".join(likely_found)

#summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
    #print(summarizer(text_for_summary, max_length=2048, min_length=30, do_sample=False))
summarize_concerns(text_for_summary)
print("##############################")
#except:
     #ZeroDivisionError
     #print("General Model Fusion fail")

### Data Model Fusion
#try:
likely_found = fusion_model_general(5)
text_for_summary = ",".join(likely_found)
#summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
summarize_concerns(text_for_summary)
print("##############################")
#except:
     #ZeroDivisionError
     #print("Data Model Fusion fail")

### Security Model Fusion
#try:
likely_found = fusion_model_general(5)
text_for_summary = ",".join(likely_found)
#summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
summarize_concerns(text_for_summary)
print("##############################")
#except:
     #ZeroDivisionError
     #print("Security Model Fusion fail")

### Privacy Model Fusion
#try:
likely_found = fusion_model_general(5)
text_for_summary = ",".join(likely_found)
#summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")
summarize_concerns(text_for_summary)
print("##############################")
#except:
     #ZeroDivisionError
     #print("Privacy Model Fail")

                   
