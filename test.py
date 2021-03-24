import numpy as np
import tensorflow as tf
import pickle
import re
from keras.models import load_model


# Your input sentence
#sentence = ['When stole book in class and the teacher caught me the rest of the class laughed at my attempt ']
#sentence=['I am so happy']
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
def remove_tags(text):
    return TAG_RE.sub('', text)
while True:
    a=input("donner le texte")

    sentence=[a]
    print("[INFO]: Loading Classes")
# Load class names
    classNames = np.load('class_names.npy')

# Load tokenizer pickle file


    print("[INFO]: Loading Tokens")
    with open('tokenizer.pickle', 'rb') as handle:

        Tokenizer = pickle.load(handle)


# Load model


    print("[INFO]: Loading Model")
    model = load_model('model_final.model')




    print("[INFO]: Preprocessing")
    # Preprocess Text
    MAX_LENGTH = maxlen = 100



    TAG_RE = re.compile(r'<[^>]+>')








# Tokenize and pad sentence
    sentence_processed = Tokenizer.texts_to_sequences(sentence)
    sentence_processed = np.array(sentence_processed)
    sentence_padded = tf.keras.preprocessing.sequence.pad_sequences(sentence_processed, padding='post', maxlen=MAX_LENGTH)



    print("""[INFO]: Prediction\n\t{}""".format(sentence[0]))
    # Get prediction for sentence
    result = model.predict(sentence_padded)
    print("-"*20)
    
    # Show prediction
    print("[INFO]: Emotion class for given text is: {}".format(classNames[np.argmax(result)]))
    





