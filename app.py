import nltk
nltk.download('popular')#  Natural Language Toolkit (NLTK) library installed will download and
# install a collection of popular NLTK data packages, including corpora,
# models etc
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random




from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


# translator pipeline for english to swahili translations
eng_swa_model_checkpoint = "Helsinki-NLP/opus-mt-en-swc"
eng_swa_tokenizer = AutoTokenizer.from_pretrained("model/eng_swa_model/")
eng_swa_model = AutoModelForSeq2SeqLM.from_pretrained("model/eng_swa_model/")

eng_swa_translator = pipeline(
    "text2text-generation",
    model=eng_swa_model,
    tokenizer=eng_swa_tokenizer,
)

def translate_text_eng_swa(text):
    translated_text = eng_swa_translator(text, max_length=128, num_beams=5)[0]['generated_text']
    return translated_text

# translator pipeline for swahili to english translations
swa_eng_model_checkpoint = "Helsinki-NLP/opus-mt-swc-en"
swa_eng_tokenizer = AutoTokenizer.from_pretrained("model/swa_eng_model/")
swa_eng_model = AutoModelForSeq2SeqLM.from_pretrained("model/swa_eng_model/")

swa_eng_translator = pipeline(
    "text2text-generation",
    model=swa_eng_model,
    tokenizer=swa_eng_tokenizer,
)

def translate_text_swa_eng(text):
    translated_text = swa_eng_translator(text, max_length=128, num_beams=5)[0]['generated_text']
    return translated_text


# Language detector
def get_lang_detector(nlp, name):
    return LanguageDetector()

# # Load the English language model
nlp = spacy.load("en_core_web_sm")

# # Register the language detection factory
Language.factory("language_detector", func=get_lang_detector)

# # Add the language detection component to the pipeline
nlp.add_pipe('language_detector', last=True)





intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    print(res)
    # language detection
    doc=nlp(res)
    detected_language = doc._.language['language']
    print(f"Detected language: {detected_language}")   

    if detected_language=="en":   
        print(translate_text_eng_swa(res))
    elif detected_language=='sw':
        print(translate_text_swa_eng(res))

    return res
    
# Creating GUI with flask
from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(userText)

    # language detection
    doc=nlp(userText)
    detected_language = doc._.language['language']
    print(f"Detected language: {detected_language}")   

    if detected_language=="en":   
        print(translate_text_eng_swa(userText))
    elif detected_language=='sw':
        print(translate_text_swa_eng(userText))

    return chatbot_response(userText)
if __name__ == "__main__":
    app.run()
