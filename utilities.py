import numpy as np
import json
import random
import torch

from os import listdir
from os.path import isfile, join

from ipywidgets import IntProgress
from ipywidgets import Dropdown
from ipywidgets import Button
from IPython.display import display

import CharRNN as crnn #See GitHub https://github.com/albertlai431/Machine-Learning/tree/master/Text%20Generation
dictionary = json.load(open("./words_dictionary.json"))

torch.cuda.empty_cache()

def readSpongebobRNN():    
    text = ""
    for season in range(1,13):
        for episode in listdir('./SpongeBob/Season ' + str(season)):
            with open("./SpongeBob/Season "+ str(season) + "/" + episode) as f:
                contents = f.read()
                contents.replace("|", ": ")

                #SPLIT SO EACH ELEMENT IS EITHER A CHARACTER OR ACTION
                contents = contents.strip().split("}} {{L|")
                contents[0] = contents[0].replace("{{L|","")
                contents[-1] = contents[-1][:contents[-1].index('}')] 

                #REMOVE ELIPSE AND ADD IN ACTION AND SENTENCE CHARS
                table = {"...":"", "''[": "<a> ", "]''": " <\\a>", "'": ""}# ".": "<\s> <s>"}
                for i, line in enumerate(contents):
                    for k, v in table.items():
                        line = line.replace(k,v)
                    contents[i] = line
                    
                text += '\n'.join(contents) + "\n"
    return text

def encodeText(text):
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text])

    return encoded

def checkDictionary(word):
    try:
        dictionary[word]
        return True
    except:
        return False