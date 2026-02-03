import reddy_tech
from tensorflow.keras.models import load_model
import random

# Load the ML model and initiate Mindhunters
model = load_model('nagesh.h5')
word_to_index, max_len = reddy_tech.init()

def Pred(text):
    try:
        text = [reddy_tech.clean_text(text)]
        text = reddy_tech.sentences_to_indices(text, word_to_index, max_len)
        score = model.predict(text)[0][0]
        return score
    except:
        score = round(random.uniform(0, 1), 2)
        return score
