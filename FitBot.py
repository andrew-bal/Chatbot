from transformers import BertForQuestionAnswering, AutoTokenizer
from transformers import pipeline

modelname = 'deepset/bert-base-cased-squad2'

model = BertForQuestionAnswering.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)


with open('chatbot_training.txt', 'r') as file:
    context = file.read().replace('\n', '')

print("\nIf you want to quit the conversation with FitBert, just type 'quit'.\n\n")
print("\t\t\t\t\tHey, I am FitBert, your personal fitness assistant and I am specialized in Crossfit and strength training.")
print("\t\t\t\t\tHow can I help you? <- FitBert")

while True:
    
    question = input("\nUser -> ")

    if question == "quit":
        break

    prediction = nlp({
        'question': question,
        'context': context
    })

    answer = prediction['answer']

    print("\n\t\t\t\t\t", answer, " <- FitBert")

print("FitBert: Thanks for the conversation, have a nice day! :)")