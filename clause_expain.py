from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# a) Get predictions



def analyze(context, question):
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    input_dict = {'context': context, 'question': question}
    result = nlp(input_dict)
    if result['answer']:
        return result
    else:
        return None
