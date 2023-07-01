import random
from clause_identifier import fusion_model_general, analysis_generate
from transformers import pipeline

# these are the files used for testing
twitter_tos = open("demo_docs/twitter_tos.txt", "r", encoding='utf-8')
text2 = twitter_tos.readlines()
facebook_tos = open("demo_docs/facebook_tos.txt", "r", encoding='utf-8')
text1 = facebook_tos.readlines()
reddit_tos = open("demo_docs/reddit_tos.txt", "r", encoding='utf-8')
text3 = reddit_tos.readlines()
youtube_tos = open("demo_docs/youtube_tos.txt", "r", encoding='utf-8')
text4 = youtube_tos.readlines()
linkedin_tos = open("demo_docs/linkedin_tos.txt", "r", encoding='utf-8')
text5 = linkedin_tos.readlines()
nytimes_tos = open("demo_docs/nytimes_tos.txt", "r", encoding='utf-8')
text6 = nytimes_tos.readlines()
openai_tos = open("demo_docs/openai_tos.txt", "r", encoding='utf-8')
text7 = openai_tos.readlines()
epic_tos = open("demo_docs/epic_tos.txt", "r", encoding='utf-8')
text8 = epic_tos.readlines()
steam_tos = open("demo_docs/steam_tos.txt", "r", encoding='utf-8')
text9 = steam_tos.readlines()
playstation_tos = open("demo_docs/playstation_tos.txt", "r", encoding='utf-8')
text11 = playstation_tos.readlines()
mississauga_tos = open("demo_docs/mississauga_tos.txt", "r", encoding='utf-8')
text12 = mississauga_tos.readlines()
ea_tos = open("demo_docs/ea_tos.txt", "r", encoding='utf-8')
text13 = ea_tos.readlines()
betterhelp_tos = open("demo_docs/betterhelp_tos.txt", "r", encoding='utf-8')
text14 = betterhelp_tos.readlines()
tiktok_tos = open("demo_docs/tiktok_tos.txt", "r", encoding='utf-8')
text15 = tiktok_tos.readlines()
netflix_tos = open("demo_docs/netflix_tos.txt", "r", encoding='utf-8')
text16 = netflix_tos.readlines()
instagram_tos = open("demo_docs/instagram_tos.txt", "r", encoding='utf-8')
text17 = instagram_tos.readlines()
twitch_tos = open("demo_docs/twitch_tos.txt", "r", encoding='utf-8')
text18 = twitch_tos.readlines()


tos_call_list = [text1, text2, text3, text4, text5, text6, text7, text8, text9,
                  text11, text12, text13, text14, text15, text16]
# assigning a random position based on a range from 0 to the length of the list
tos_call_text = tos_call_list[random.choice(range(0, len(tos_call_list)))]

summary_text_list = []

### General Model Fusion
file_general = open("output_general.txt", "w")
classes_general = ['Concerning Clause for User','Non-Concerning Clause for User']
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_general[0]

likely_found = fusion_model_general(text18, 5, classes_general, file_general, 0.7,0.6)
text_for_summary = ",".join(likely_found)

summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
   
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_general[0])
    explained_text.append(analyze_text)
file_general.write(str(explained_text))
file_general.close()

print("##############################")


### Data Model Fusion
classes_data = ['Potential Data Use Concern','No Data Use Concern']
file_data = open("output_data.txt", "w")
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_data[0]

likely_found = fusion_model_general(text18, 5, classes_data, file_data,0.8,0.5)
text_for_summary = ",".join(likely_found)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_data[0])
    explained_text.append(analyze_text)
file_data.write(str(explained_text))
file_data.close()

print("##############################")


### Security Model Fusion
classes_security = ['Potential Security Concern for User','No Security Concern for User']
file_security = open("output_security.txt", "w")
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_security[0]

likely_found = fusion_model_general(text18, 5,classes_security,file_security,0.75,0.5)
text_for_summary = ",".join(likely_found)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_security[0])
    explained_text.append(analyze_text)
file_security.write(str(explained_text))
file_security.close()

print("##############################")


### Privacy Model Fusion
classes_privacy = ['Potential Privacy Concern','No Privacy Concern ']
file_privacy = open("output_privacy.txt", "w")
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_privacy[0]

likely_found = fusion_model_general(text18, 5, classes_privacy, file_privacy,0.8,0.5)
text_for_summary = ",".join(likely_found)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_privacy[0])
    explained_text.append(analyze_text)
file_privacy.write(str(explained_text))
file_privacy.close()

print("##############################")

### Legal Model Fusion
classes_legal = ['Potential Legal Concern','No Legal Concern ']
file_legal = open("output_legal.txt", "w")
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_legal[0]

likely_found = fusion_model_general(text18, 5, classes_legal, file_legal,0.8,0.5)
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_legal[0])
    explained_text.append(analyze_text)
file_legal.write(str(explained_text))
file_legal.close()

text_for_summary = ",".join(likely_found)

print("##############################")

### Versitility Function (Currently Streamers)
classes_streamer = ['Potential Concern for Streamers','No Concern for Streamers']
file_streamer = open("output_streamers.txt", "w")
analyze_prompt = f"Instruction: Explain why this text found from a TOS/EULA/Privacy Policy was labelled from a natural lanugage model as " + classes_privacy[0]

likely_found = fusion_model_general(text18, 5, classes_streamer, file_streamer,0.8,0.5)
text_for_summary = ",".join(likely_found)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
explained_text = []
for x in likely_found:
    analyze_text = analysis_generate(x,classes_privacy[0])
    explained_text.append(analyze_text)
file_privacy.write(str(explained_text))
file_privacy.close()

print("##############################")



summary_output = open("output_summary.txt", "w")
print(summary_text_list)

