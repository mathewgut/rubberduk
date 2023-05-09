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
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel, pipeline, XLNetTokenizer, XLNetForSequenceClassification
import time
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy
from torch import nn
from torch.optim import AdamW
import re
from clause_audit import audit_concerning_clauses
# this was to limit cuda ram usage, but I am having a lot of issues getting it to set max vram, so its disabled for now
#import os

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:5000"



time_current = time.asctime(time.localtime(time.time()))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# model configuration, change the directory to just a name to create a new folder for a new model (eg.config = BertConfig.from_json_file("New Model")) 
config = BertConfig.from_json_file("Model Data\\config.json")

# Initialize the model with the configuration
model = BertForSequenceClassification(config)

# Load the pre-trained weights
# Only use if a pre trained model already exists. The weights are the importance of certain attributes it finds in a neural network.
model.load_state_dict(torch.load("Model Data\\pytorch_model.bin"))

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
tiktok_tos = open("tiktok_tos.txt", "r", encoding='utf-8')
text10 = tiktok_tos.read()
playstation_tos = open("playstation_tos.txt", "r", encoding='utf-8')
text11 = playstation_tos.read()
mississauga_tos = open("mississauga_tos.txt", "r", encoding='utf-8')
text12 = mississauga_tos.read()
ea_tos = open("ea_tos.txt", "r", encoding='utf-8')
text13 = ea_tos.read()
betterhelp_tos = open("betterhelp_tos.txt", "r", encoding='utf-8')
text14 = betterhelp_tos.read()

tos_call_list = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, text13, text14]
tos_call_text = random.choice(tos_call_list)
device = torch.device('cpu')  # This is set to just CPU. you can change it by putting GPU instead. The GPU will allow for immensley faster training, but there is no current way to limit VRAM (at least that works)

concerning_list = [
        "We may collect and use your personal information for marketing purposes.",
        "We reserve the right to share your personal information with third parties.",
        "We may use cookies to track your online activity.",
        "We may sell your personal information to third parties.",
        "We may share your personal information with our affiliates and partners for their own marketing purposes.",
        "We may use your personal information to build a profile about you and target you with personalized advertising.",
        "We may collect and use your personal information for legitimate business purposes as allowed by law.",
        "By using our services, you agree to arbitrate any disputes rather than going to court or participating in class actions.",
        "We may modify this policy at any time and without notice, and your continued use of our services constitutes acceptance of the changes.",
        "Your use of our services may be subject to additional terms and conditions, such as community guidelines or payment terms.",
        "We may disclose your personal information to law enforcement or government authorities as required by law or in response to a legal request.",
        "We may collect and store your IP address and device information when you use our services.",
        "We may share your personal information with our subsidiaries and affiliates, including those located outside your country of residence.",
        "We may use third-party service providers to process your personal information, and they may have access to your information.",
        "We may retain your personal information even after you delete your account or stop using our services.",
        "We may use your personal information for research and analysis, including to develop new products or services.",
        "We may use your personal information to contact you with marketing materials, including by email, SMS, or push notifications.",
        "We may share your personal information with third-party advertisers to show you personalized ads based on your interests and activity.",
        "We may use cookies and other tracking technologies to collect information about your browsing behavior and online activity.",
        "We may disclose your personal information to third parties in the event of a merger, acquisition, or bankruptcy.",
        "We may transfer your personal information to countries that do not have the same data protection laws as your country of residence.", 
        "Your personal information may be used to create a detailed profile of your online behavior and preferences.",
        "We may track your location and collect geolocation data to target you with location-based advertisements.",
        "We may use facial recognition technology to identify and track individuals in photos or videos.",
        "We may collect data from your social media accounts and use it for targeted advertising.",
        "We may share your personal information with data brokers or other third parties for marketing purposes.",
        "We may use machine learning algorithms to analyze your personal information and make automated decisions.",
        "Your personal information may be stored and processed on servers located in different countries with varying data protection standards.",
        "We may use your personal information to infer sensitive details about your political opinions, religious beliefs, or sexual orientation.",
        "We may collect and analyze your communication data, including emails and chat conversations, for targeted advertising or data mining purposes.",
        "We may use device fingerprinting techniques to track and identify your devices across multiple websites and applications.",
        "We may collect and store data about your interactions with our customer support, including transcripts of conversations and support tickets.",
        "Your personal information may be shared with advertisers and marketing companies to create custom audience segments for targeted advertising.",
        "We may collect and analyze data from wearable devices or smart home devices to gain insights into your lifestyle and behavior patterns.",
        "We may use cross-device tracking technologies to link your activities across different devices and deliver personalized content.",
        "We may collect and process biometric data, such as fingerprints or voiceprints, for authentication or identification purposes.",
        "We may use personal information collected from children for targeted advertising or profiling purposes.",
        "We may share your personal information with business partners or affiliates for their own marketing or advertising purposes.",
        "Your personal information may be disclosed to intelligence agencies or government authorities for surveillance or national security purposes.",
        "We may use your personal information to conduct A/B testing or experiments without your explicit consent.",
        "We may collect and analyze data from public sources or data aggregators to supplement or enhance your personal profile.",
        "We may use session replay technologies to record and analyze your interactions with our website or application.",
        "Your personal information may be sold as part of a data package or bundled with other user data for commercial purposes.",
        "Your personal information may be shared with select business partners to offer you tailored promotions and exclusive offers based on your interests and preferences.",
        "We reserve the right to collect and analyze your data, including browsing history and device information, to deliver targeted advertisements and optimize our services.",
        "By using our platform, you agree that we may disclose your personal information to third-party service providers located in other countries, subject to their data protection laws.",
        "We may utilize advanced data analytics techniques to extract insights from your personal information and create profiles for targeted marketing campaigns.",
        "Your personal information may be transferred and stored on servers located outside your country of residence, where data protection laws may differ.",
        "You hereby grant us a perpetual, irrevocable, worldwide, royalty-free license to use, reproduce, modify, adapt, publish, translate, create derivative works from, distribute, and display your user-generated content in any medium and for any purpose, including commercial exploitation.",
        "We may share your personal information with our affiliates, subsidiaries, and third-party partners, including data processors, in accordance with applicable data protection laws and for the purposes of targeted advertising and direct marketing.",
        "In the event of a merger, acquisition, or bankruptcy, your personal information may be transferred as a business asset to the acquiring entity or its successors, subject to their assumption of the obligations set forth in this privacy policy.",
        "By using our services, you acknowledge and agree that we may collect and process your sensitive personal data, including information about your health, religious beliefs, and biometric data, for the purposes of providing personalized services and targeted advertising.",
        "We retain the right to amend, modify, or terminate this agreement at any time, without prior notice, and your continued use of the services constitutes acceptance of the revised terms, including any changes to the collection, use, and sharing of your personal information.",
        "We may use your personal information to create a detailed profile of your online behavior and preferences.",
        "We may track your location and collect geolocation data to target you with location-based advertisements.",
        "We may use facial recognition technology to identify and track individuals in photos or videos.",
        "We may collect data from your social media accounts and use it for targeted advertising.",
        "We may share your personal information with data brokers or other third parties for marketing purposes.",
        "We may use machine learning algorithms to analyze your personal information and make automated decisions.",
        "We may use your personal information to infer sensitive details about your political opinions, religious beliefs, or sexual orientation.",
        "We may collect and analyze your communication data, including emails and chat conversations, for targeted advertising or data mining purposes.",
        "We may use device fingerprinting techniques to track and identify your devices across multiple websites and applications.",
        "We may collect and store data about your interactions with our customer support, including transcripts of conversations and support tickets.",
        "We may share your personal information with advertisers and marketing companies to create custom audience segments for targeted advertising.",
        "We may collect and analyze data from wearable devices or smart home devices to gain insights into your lifestyle and behavior patterns.",
        "We may use cross-device tracking technologies to link your activities across different devices and deliver personalized content.",
        "We may collect and process biometric data, such as fingerprints or voiceprints, for authentication or identification purposes.",
        "We may use personal information collected from children for targeted advertising or profiling purposes.",
        "By accepting these terms, you agree to waive your right to pursue any legal claims through class actions and instead agree to binding arbitration.",
        "We reserve the right to disclose your personal information to third parties as required by law or to comply with a valid legal process.",
        "You acknowledge and agree that any disputes arising from or related to these terms shall be subject to the exclusive jurisdiction of the courts in [jurisdiction].",
        "We may amend this agreement at any time without notice, and your continued use of the services shall constitute acceptance of the modified terms.",
        "You grant us an irrevocable and perpetual license to use, reproduce, distribute, and display your user-generated content, including any intellectual property rights therein.",
        "We may suspend or terminate your access to the services at any time and without liability, including for any violation of these terms or our policies.",
        "We provide the services 'as is' and make no warranties or representations regarding their accuracy, reliability, or fitness for a particular purpose.",
        "You agree to indemnify and hold us harmless from any claims, damages, or liabilities arising out of your use of the services or violation of these terms.",
        "We shall not be liable for any direct, indirect, incidental, consequential, or punitive damages, regardless of the cause of action, even if advised of the possibility of such damages.",
        "The services may contain links to third-party websites or resources, and we are not responsible for the availability or content of such external sites.",
        "We may assign or transfer our rights and obligations under these terms to any third party without your consent.",
        "You acknowledge that the services may be subject to technical limitations or interruptions, and we do not guarantee continuous, error-free access to the services.",
        "You agree to defend, indemnify, and hold us harmless from any claims, losses, or damages arising from your breach of any representation or warranty in these terms.",
        "We make no representations or warranties regarding the accuracy or completeness of any information provided through the services, and you use such information at your own risk.",
        "To the fullest extent permitted by law, our liability for any claim arising out of or in connection with these terms or the services shall be limited to the amount you paid, if any, for the use of the services.",
        "By accepting these terms, you grant us an irrevocable, worldwide, royalty-free license to use, reproduce, modify, adapt, publish, translate, create derivative works from, distribute, and display your user-generated content in any medium and for any purpose, including commercial exploitation.",
        "We may disclose your personal information to third parties as required by law or to comply with a valid legal process, without providing notice to you.",
        "You acknowledge and agree that any disputes arising from or related to these terms shall be subject to the exclusive jurisdiction of the courts in [jurisdiction], limiting your options for legal recourse.",
        "We reserve the right to amend this agreement at any time without notice, and your continued use of the services constitutes acceptance of the modified terms, including any changes to the collection, use, and sharing of your personal information.",
        "You acknowledge and agree that we may collect and process your sensitive personal data, including information about your health, religious beliefs, and biometric data, for the purposes of providing personalized services and targeted advertising.",
        "We may utilize advanced data analytics techniques to extract insights from your personal information and create profiles for targeted marketing campaigns, potentially leading to extensive profiling.",
        "Your personal information may be transferred and stored on servers located outside your country of residence, where data protection laws may differ, potentially exposing your data to lower privacy standards.",
        "By using our platform, you agree that we may disclose your personal information to third-party service providers located in other countries, subject to their data protection laws, which may not provide the same level of protection as your country of residence.",
        "We may share your personal information with our affiliates, subsidiaries, and third-party partners, including data processors, in accordance with applicable data protection laws and for the purposes of targeted advertising and direct marketing, potentially resulting in extensive data sharing.",
        "We reserve the right to collect and analyze your data, including browsing history and device information, to deliver targeted advertisements and optimize our services, potentially leading to intrusive tracking and profiling.",
        "We reserve the right to change these terms at any time, without notice.",
        "We may collect your name, email address, and phone number, and use it for marketing purposes.",
        "We may share your personal information with third-party partners.",
        "We are not responsible for any damages caused by our service.",
        "We may terminate your account at any time, for any reason.",
        "We may sell your personal information to third-party partners.",
        "We may use your personal information for any purpose we see fit.",
        "You have no rights to your personal information.",
        "We may track your activity on our website and use that information to target you with advertising.",
        "We may use your personal information to make decisions about you, such as whether to approve your loan application.",
        "We may use your personal information to sell it to other companies.",
        "We may use your personal information to contact you with marketing messages.",
        "We may use your personal information to track your location.",
        "We may use your personal information to monitor your behavior.",
        "We may use your personal information to sell it to the government.",
        "We may use your personal information to harm you in any way we see fit.",
        "We may use your personal information to sell it to our competitors.",
        "We may use your personal information to create a profile of you.",
        "We may use your personal information to make decisions about you without your knowledge or consent.",
        "We may use your personal information to discriminate against you.",
        "We may use your personal information to invade your privacy.",
        "We may use your personal information to harm you in any way we see fit.",
    
        # Legalese
        "We reserve the right to modify these Terms at any time, without notice, and your continued use of the Services following such modification constitutes your acceptance of the modified Terms.",
        "We may collect, use, and disclose your personal information in accordance with our Privacy Policy.",
        "We are not responsible for any damages caused by your use of the Services, including, but not limited to, direct, indirect, incidental, consequential, or punitive damages.",
        "You agree to indemnify and hold us harmless from any and all claims, liabilities, damages, and expenses, including reasonable attorneys' fees, arising out of or in connection with your use of the Services.",
        
        # Plain English
        "We can change the rules whenever we want, and we don't have to tell you.",
        "We can take your personal information and use it for anything we want.",
        "If something goes wrong, you're on your own.",
        "If you sue us, you have to pay our legal fees.",
        
        # Combination of Legalese and Plain English
        "We reserve the right to modify these Terms at any time, without notice. However, we will always try to give you as much notice as possible.",
        "We may collect your personal information and use it for marketing purposes. However, you can always opt out of marketing communications.",
        "We are not responsible for any damages caused by our service, but we will always try to make things right."
        "We reserve the right to monitor all communications and activities on our platform, and we may disclose any information we collect to law enforcement or other government agencies if we believe that it is necessary to do so.",
        "We may use your personal information for marketing purposes, even if you have opted out of marketing communications.",
        "We may sell your personal information to third-party companies.",
        
        # Plain English
        "We can read your messages and see what you're doing on our platform.",
        "We can give your information to the government.",
        "We can sell your information to other companies.",
        
        # Combination of Legalese and Plain English
        "We reserve the right to monitor all communications and activities on our platform, but we will only do so if we believe that it is necessary to protect our users or to comply with the law. We will not disclose any information we collect to law enforcement or other government agencies without a warrant or other legal process. We may use your personal information for marketing purposes, but you can always opt out of marketing communications. We may sell your personal information to third-party companies, but we will only do so in accordance with our privacy policy.",
        
        # Other concerning clauses
        "We may change these Terms at any time, without notice.",
        "We may suspend or terminate your account for any reason, at our sole discretion.",
        "These Terms are governed by and construed in accordance with the laws of the State of California, without regard to its conflict of laws provisions.",
        "Any dispute arising out of or relating to these Terms shall be resolved by binding arbitration in accordance with the rules of the American Arbitration Association, except that you may assert individual claims in small claims court if they qualify.",
        "The arbitrator's award shall be final and binding, and may be entered as a judgment in any court of competent jurisdiction.",
        "You agree to waive any right to a jury trial in any action or proceeding arising out of or relating to these Terms.",
        "These Terms constitute the entire agreement between you and us with respect to your use of the Services, and they supersede all prior or contemporaneous communications and proposals, whether oral or written, between you and us with respect to the Services.",
        "We may assign these Terms to any third party at any time. You may not assign these Terms or your rights under these Terms to any third party without our prior written consent.",
        "If any provision of these Terms is held to be invalid or unenforceable, such provision shall be struck from these Terms and the remaining provisions shall remain in full force and effect.",
        "Our failure to enforce any provision of these Terms shall not be construed as a waiver of such provision.",
        "These Terms are binding upon and inure to the benefit of the parties hereto and their respective successors and permitted assigns.",
        
        # Other concerning statements
        "We reserve the right to change our prices at any time, without notice.",
        "We may cancel your account if you violate our Terms of Service.",
        "We may use your personal information for marketing purposes, even if you have opted out of marketing communications.",
        "We may sell your personal information to third-party companies.",
        "We may use cookies and other tracking technologies to collect information about your use of our Services.",

        "We may use your personal information to contact you about our Services.",
        "We may use your personal information to comply with the law.",
        "We may use your personal information to protect our rights.",
        "We reserve the right to change these terms at any time without notice.",

        # These statements give the company a lot of power and flexibility, and they may not be in the best interests of the user.

        "We reserve the right to change these terms at any time without notice.",
        "We may terminate your account at any time for any reason.",
        "We are not responsible for any damages that you may incur as a result of using our services.",
        "We may use your personal information for any purpose, including marketing, advertising, and research.",
        "We may share your personal information with our affiliates and partners.",
        "We may use cookies and other tracking technologies to collect information about your use of our services.",

        # These statements are vague and ambiguous, and they make it difficult for the user to understand what they are agreeing to.

        "We may collect, use, and share your personal information in any way we see fit.",
        "We may make changes to these terms at any time, and your continued use of our services constitutes your acceptance of those changes.",
        "We are not liable for any damages that you may incur as a result of using our services, even if those damages are caused by our negligence.",

        # These statements are misleading or deceptive, and they may not be accurate.

        "We are the sole owners of all intellectual property rights in our services, including our website, software, and content.",
        "You agree not to use our services for any illegal or unauthorized purpose.",
        "You agree not to violate our terms of service in any way.",
        # Formal
        "We may use your personal information in ways that are not specifically described in this policy.",
        "We may disclose your personal information to third-party partners for any purpose.",
        "We may use your personal information for any purpose, at any time, without notice.",

        # Informal
        "We may do whatever we want with your data.",
        "We may share your data with anyone we want.",
        "We may use your data for any reason we want.",

        # Vague
        "We may use your personal information in ways that are not specifically prohibited by this policy.",
        "We may disclose your personal information to third-party partners in accordance with applicable law.",
        "We may use your personal information for any purpose that is reasonably related to the services we provide.",

        # Formal
        "We may use cookies and other tracking technologies to collect information about your use of our services, including your IP address, browser type, operating system, referring website, and pages you visit on our site. We may also use this information to track your activity across other websites and online services. This information may be used to target advertising, to improve our services, and to protect our rights and the rights of others.",

        "We may share your personal information with third-party partners, including advertisers, for the purpose of cross-site tracking. This means that these third-party partners may track your activity across our website and other websites in order to target advertising to you.",

        # Informal
        "We can track you across the internet and use your data to target ads at you.",
        "We can share your data with other companies so they can track you too.",

        # Formal
        "We may use your personal information in any way we see fit, without your consent.",
        "We may share your personal information with third-party partners, without your knowledge or consent.",
        "We may keep your personal information indefinitely, even if you no longer use our services.",

        # Informal
        "We can do whatever we want with your data.",
        "We can share your data with anyone we want.",
        "We can keep your data forever.",

        # Vague clauses

        # Formal
        "We may use your personal information for any purpose that is reasonably related to the services we provide.",
        "We may disclose your personal information to third-party partners in accordance with applicable law.",
        "We may use your personal information for research purposes.",

        # Informal
        "We may use your data for stuff.",
        "We may share your data with other companies.",
        "We may use your data to figure out what you like.",

        # Formal
        "We may use your personal information in any way we see fit, without your consent, including but not limited to: selling it to third-party partners, using it to target advertising, and sharing it with government agencies.",
        "We may share your personal information with third-party partners, without your knowledge or consent, for the purpose of marketing their products and services to you. You can opt out of this sharing by visiting the following website: [website address].",
        "We may keep your personal information indefinitely, even if you no longer use our services, and we may use it for any purpose that we see fit.",

        # Informal
        "We can do whatever we want with your data, and we don't need your permission.",
        "We can share your data with anyone we want, and we don't have to tell you.",
        "We can keep your data forever, and we can use it for anything we want.",

        """

        We may use your personal information in any manner we deem necessary or appropriate, in our sole discretion, including but not limited to:

        Storing it,
        Analyzing it,
        Combining it with other information,
        Selling it to third parties,
        Using it to create profiles of your interests and preferences,
        Targeting you with advertising,
        Sharing it with government agencies,
        And/or otherwise using it in any way we see fit.
        
        We may also change this policy at any time, without notice.
        """,

        
        # Technical talk
        """
        We may use your personal information to provide, maintain, and improve our services, to develop new services, to provide customer support, to protect our rights and the rights of others, to comply with applicable law, and for any other purpose that we deem necessary or appropriate.

        We may also share your personal information with third parties, including our affiliates, partners, and service providers, for the purposes of providing, maintaining, and improving our services, developing new services, providing customer support, protecting our rights and the rights of others, complying with applicable law, and for any other purpose that we deem necessary or appropriate.

        We may also use your personal information to create profiles of your interests and preferences, and to target you with advertising.

        We may also change this policy at any time, without notice.
        """,
        
        ]

not_concerning = [
        "We may collect your personal information to provide you with better services.",
        "We will not share your personal information with third parties without your consent.",
        "We may use cookies to improve the functionality of our website.",
        "We will only collect the personal information that is necessary to provide you with our services.",
        "We take appropriate measures to protect your personal information from unauthorized access or disclosure.",
        "We may use your personal information to respond to your inquiries or requests.",
        "We are committed to protecting your privacy and maintaining the security of your personal information.",
        "We will only use your personal information for the purpose for which it was collected.",
        "We may use your personal information to personalize your experience with our services.",
        "We may anonymize or aggregate your personal information for research or statistical purposes.",
        "We will not sell or rent your personal information to third parties without your explicit consent.",
        "We will only retain your personal information for as long as necessary to provide you with our services or as required by law.",
        "We may use your personal information to improve our services and develop new features.",
        "We will notify you of any changes to our privacy policy and give you the opportunity to opt-out of any material changes.",
        "We will provide you with access to your personal information and allow you to correct or delete it if necessary.",
        "We will not discriminate against you for exercising your privacy rights.",
        "We will never sell your personal information without your consent.",
        "We take the security of your personal information seriously and use appropriate measures to protect it.",
        "We only collect the personal information that is necessary to provide you with our services.",
        "Your personal information is used solely to respond to your inquiries or requests.",
        "We are committed to providing you with transparent information about our data practices.",
        "We will always respect your privacy and will not share your personal information with third parties without your consent.",
        "We may collect anonymous usage data to analyze and improve our services.",
        "We may use your personal information to send you updates and notifications about our products or services.",
        "We may share your personal information with trusted partners who assist us in delivering our services.",
        "We may use your personal information to customize and tailor the content you see on our platform.",
        "We will obtain your consent before using your personal information for any purpose not disclosed in our privacy policy.",
        "We may collect and store your personal information securely in encrypted form.",
        "We may use your personal information to verify your identity and prevent fraudulent activities.",
        "We will provide you with options to manage your privacy settings and control the information you share.",
        "We may use your personal information to personalize advertisements or promotional offers.",
        "We will promptly address any data breaches or security incidents to protect your personal information.",
        "We may use your personal information to conduct internal research and analysis to improve our services.",
        "We will only share your personal information with third parties who are bound by confidentiality obligations.",
        "We may collect and use your personal information in accordance with applicable data protection laws.",
        "We will delete or anonymize your personal information when it is no longer needed for the purposes stated.",
        "We may use your personal information to communicate important updates or changes to our services.",
        "We will provide you with clear and accessible information about how your personal information is used and protected.",
        "We will not use your personal information in ways that are incompatible with the purposes for which it was collected.",
        "We may aggregate and analyze anonymous data to gain insights and improve our products or services.",
        "We will respect your communication preferences and honor your choices regarding the use of your personal information.",
        "We may use industry-standard security measures to safeguard your personal information from unauthorized access.",
        "We will not retain your personal information longer than necessary for the purposes it was collected.",
        "Your personal information will be handled in compliance with applicable privacy laws and regulations.",
        "We may use your personal information to provide you with relevant recommendations and suggestions.",
        "We will not disclose your personal information to third parties unless required by law or with your explicit consent.",
        "We may use data encryption technologies to protect the confidentiality of your personal information.",
        "We will only use your email address to send you important service-related notifications and updates about your account.",
        "Your personal information will not be shared with any third parties without your explicit consent.",
        "We employ strict security measures to protect your personal information from unauthorized access, ensuring its confidentiality and integrity.",
        "We may collect anonymized usage data to analyze trends and improve our services, but this information cannot be used to identify you personally.",
        "You have the right to access, modify, and delete your personal information from our records at any time.",
        "We will only use your personal information to fulfill the purposes for which it was collected, as outlined in this privacy policy, and in compliance with applicable data protection laws.",
        "We may process your personal data based on your consent, which you can withdraw at any time, and we will not process your personal data for any other purposes without obtaining your explicit consent.",
        "We employ reasonable technical and organizational measures to safeguard your personal information against unauthorized access, loss, or alteration and to ensure its confidentiality and integrity.",
        "Your personal information will not be disclosed to third parties unless required by law or with your explicit consent, and we will always strive to provide you with clear and transparent information about any data sharing practices.",
        "You have the right to request access to your personal information, rectify any inaccuracies, and have your data deleted or anonymized in compliance with applicable data protection regulations.",
        "We will only collect the personal information that is necessary to provide you with our services.",
        "We take appropriate measures to protect your personal information from unauthorized access or disclosure.",
        "We may use your personal information to respond to your inquiries or requests.",
        "We will only use your personal information for the purpose for which it was collected.",
        "We will not sell or rent your personal information to third parties without your explicit consent.",
        "We may use your personal information to improve our services and develop new features.",
        "We will notify you of any changes to our privacy policy and give you the opportunity to opt-out of any material changes.",
        "We will provide you with access to your personal information and allow you to correct or delete it if necessary.",
        "We will not discriminate against you for exercising your privacy rights.",
        "We take the security of your personal information seriously and use appropriate measures to protect it.",
        "We may collect anonymous usage data to analyze and improve our services.",
        "We may use your personal information to send you updates and notifications about our products or services.",
        "We may share your personal information with trusted partners who assist us in delivering our services.",
        "We may use your personal information to customize and tailor the content you see on our platform.",
        "We will obtain your consent before using your personal information for any purpose not disclosed in our privacy policy.",
        "This agreement constitutes the entire understanding between the parties and supersedes any prior agreements or representations, whether oral or written.",
        "Any dispute arising out of or relating to this agreement shall be subject to the exclusive jurisdiction of the courts in [Jurisdiction].",
        "The failure to enforce any provision of this agreement shall not be construed as a waiver of that provision or the right to enforce it in the future.",
        "If any provision of this agreement is held to be invalid or unenforceable, the remaining provisions shall continue in full force and effect.",
        "This agreement shall be binding upon and inure to the benefit of the parties and their respective successors and assigns.",
        "The headings and captions in this agreement are for convenience only and shall not affect the interpretation or construction of any provision.",
        "No agency, partnership, joint venture, or employment relationship is created as a result of this agreement, and neither party has the authority to bind the other.",
        "The rights and remedies provided in this agreement are cumulative and not exclusive, and the exercise of one shall not preclude the exercise of any other.",
        "Any notices required or permitted under this agreement shall be in writing and deemed effectively given upon delivery if delivered in person, by overnight courier, or by registered mail.",
        "This agreement may be executed in counterparts, each of which shall be deemed an original, but all of which together shall constitute one and the same instrument.",
        "We employ advanced encryption and security measures to protect your personal information from unauthorized access or disclosure.",
        "We have strict access controls in place to ensure that only authorized personnel have access to your personal information.",
        "We regularly update our systems and infrastructure to keep your data safe from emerging security threats.",
        "We conduct regular security audits and assessments to identify and address any potential vulnerabilities in our systems.",
        "We have implemented robust data protection policies and procedures to ensure compliance with applicable data protection laws.",
        "We provide training to our employees to ensure they understand the importance of data security and the proper handling of personal information.",
        "We have a dedicated team that monitors and responds to any potential data breaches or security incidents.",
        "We conduct regular backups of your data to ensure its availability and integrity in case of any unforeseen events.",
        "We adhere to industry best practices and standards when it comes to data security and privacy.",
        "We agree to provide you with the Services in accordance with the Terms.",
        "You agree to pay for the Services in accordance with the Payment Terms.",
        "We may terminate your use of the Services if you violate the Terms.",
        
        # Legalese
        "We agree to use reasonable efforts to provide the Services in a timely and efficient manner.",
        "We will not be liable for any damages caused by circumstances beyond our reasonable control.",
        "We may assign our rights and obligations under this Agreement to another party.",
        
        # Plain English
        "We will try our best to provide the services to you on time and efficiently.",
        "We are not responsible for any damages caused by things that we cannot control.",
        "We can transfer our rights and obligations to another company.",
        
        # Combination of Legalese and Plain English
        "We agree to use reasonable efforts to provide the Services in a timely and efficient manner. However, we cannot guarantee that the Services will always be available or that they will be free from errors.",
        "We will not be liable for any damages caused by circumstances beyond our reasonable control, such as acts of God, war, or terrorism.",
        "We may assign our rights and obligations under this Agreement to another party, but we will always notify you of any such assignment.",
        
        # Legalese about personal information
        "We agree to use reasonable efforts to protect your personal information.",
        "We will not sell or share your personal information with third parties without your consent.",
        "You have the right to access, correct, or delete your personal information.",
        
        # Plain English about personal information
        "We will try our best to keep your personal information safe.",
        "We will not sell or give your personal information to other companies.",
        "You can ask us to see, change, or delete your personal information.",
        
        # Combination of Legalese and Plain English about personal information
        "We agree to use reasonable efforts to protect your personal information. However, we cannot guarantee that your personal information will always be secure. If you believe that your personal information has been compromised, please contact us immediately.",
        "We will not sell or share your personal information with third parties without your consent. However, we may share your personal information with third-party service providers who help us to provide the Services, such as payment processors or customer support providers. These third-party service providers are bound by confidentiality agreements and are not allowed to use your personal information for any other purpose.",
        "You have the right to access, correct, or delete your personal information. You can exercise these rights by contacting us at [email protected] or [phone number].",
        
        # Other non-concerning clauses
        "We reserve the right to modify these Terms at any time, without notice.",
        "We may suspend or terminate your use of the Services for any reason, at our sole discretion.",
        "These Terms are governed by and construed in accordance with the laws of the State of California, without regard to its conflict of laws provisions.",
        "Any dispute arising out of or relating to these Terms shall be resolved by binding arbitration in accordance with the rules of the American Arbitration Association, except that you may assert individual claims in small claims court if they qualify.",
        "The arbitrator's award shall be final and binding, and may be entered as a judgment in any court of competent jurisdiction.",
        "You agree to waive any right to a jury trial in any action or proceeding arising out of or relating to these Terms.",
        "These Terms constitute the entire agreement between you and us with respect to your use of the Services, and they supersede all prior or contemporaneous communications and proposals, whether oral or written, between you and us with respect to the Services.",
        "We may assign these Terms to any third party at any time. You may not assign these Terms or your rights under these Terms to any third party without our prior written consent.",
        "If any provision of these Terms is held to be invalid or unenforceable, such provision shall be struck from these Terms and the remaining provisions shall remain in full force and effect.",
        "Our failure to enforce any provision of these Terms shall not be construed as a waiver of such provision.",
        "These Terms are binding upon and inure to the benefit of the parties hereto and their respective successors and permitted assigns.",
        # These statements are clear, concise, and easy to understand.

        "We will use your personal information to provide you with the services that you have requested.",
        "We will not share your personal information with third parties without your consent.",
        "We will protect your personal information in accordance with applicable law.",
        "You have the right to access, correct, or delete your personal information.",
        "You have the right to opt out of marketing communications.",
        "You have the right to file a complaint with the appropriate regulatory authority.",

        # These statements are specific and unambiguous, and they make it easy for the user to understand what they are agreeing to.

        "We will only collect your personal information that is necessary to provide you with the services that you have requested.",
        "We will only use your personal information for the purposes that you have consented to.",
        "We will take reasonable steps to protect your personal information from unauthorized access, use, or disclosure.",
        "You have the right to withdraw your consent at any time.",
        "You have the right to complain to the appropriate regulatory authority if you believe that your personal information has been misused.",

        # These statements are accurate and truthful.

        "We are not the sole owners of all intellectual property rights in our services.",
        "You may use our services for any legal purpose.",
        "You agree to comply with our terms of service.",
        # Formal
        "We will only use your personal information for the purposes stated in this policy.",
        "We will only disclose your personal information to third-party partners with your consent.",
        "We will only use your personal information for a reasonable period of time, after which it will be deleted.",

        # Informal
        "We will only use your data for the things we told you we would use it for.",
        "We will only share your data with people you have approved of.",
        "We will only use your data for as long as we need it.",

        # Formal
        "We may use your personal information in ways that are not specifically described in this policy, but only for purposes that are reasonably related to the services we provide.",
        "We may disclose your personal information to third-party partners, but only with your consent.",
        "We will only use your personal information for a reasonable period of time, after which it will be deleted.",

        # Informal
        "We may use your data for things we haven't told you about yet, but only if it's related to the services we provide.",
        "We may share your data with other companies, but only if you agree to it.",
        "We will only keep your data for as long as we need it, and then we will delete it.",

        "We will only use your personal information for the purposes stated in this policy and in accordance with applicable law. We will not sell, trade, or rent your personal information to third-party partners without your consent.",
        "You have the right to access your personal information and to correct any inaccuracies. You can also opt out of data collection at any time. For more information, please see our privacy policy.",
        "We use cookies and other tracking technologies to collect information about your use of our services. This information is used to improve our services and to target advertising. You can learn more about our use of cookies and other tracking technologies in our privacy policy.",

        # Informal
        "We will only use your data for the things we told you we would use it for.",
        "You can always ask us to see your data and to make sure it's correct.",
        "You can always tell us if you don't want us to collect your data.",

        # Non-concerning clauses

        "We will only use your personal information for the following purposes: To provide our services to you To communicate with you: To improve our services, To protect our rights and the rights of others, To comply with applicable law. We will not sell, trade, or rent your personal information to third-party partners without your consent. You can learn more about our commitment to privacy in our privacy policy.",

        "You have the following rights with respect to your personal information: The right to access your personal information, The right to correct any inaccuracies in your personal information, The right to opt out of data collection at any time, The right to request that we delete your personal information. You can exercise these rights by contacting us at [email address]. You can learn more about your rights in our privacy policy.",

        "We use cookies and other tracking technologies to collect information about your use of our services. This information is used to improve our services and to target advertising. You can learn more about our use of cookies and other tracking technologies in our privacy policy.",



    ]

train_data = []

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

    model.to(device)
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

    return concerning_clauses

# Randomly select a TOS text
tos_call_text = random.choice(tos_call_list)

# Example usage
concerning_clauses = extract_concerning_clauses(tos_call_text, window_size=3)
for clause in concerning_clauses:
    print(clause)




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

model.save_pretrained("Model Data")

##### XLNET ######


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

"""
f1_score = evaluate_audit(audit_model, eval_dataloader)
print("F1 Score:", f1_score)
"""
# Function to audit the concerning clauses




results = audit_concerning_clauses(concerning_clauses)
# Print the audit results
for clause, predicted_label in results:
    if predicted_label == 1:
        print(f"Concerning clause: {clause}")
    else:
        print(f"Not concerning clause: {clause}")



# Close the file handles
for tos_file in [twitter_tos, facebook_tos, reddit_tos, youtube_tos, linkedin_tos, nytimes_tos, openai_tos, epic_tos, steam_tos, tiktok_tos]:
    tos_file.close()

