from collections import defaultdict, Counter
import numpy as np
import random
import bisect
import re

class sentence_model:
    def __init__(self, character, ngram=2, alpha=.1, smooth=False):
        self.ngram = ngram
        self.ngramCounts = [i for i in range(ngram - 1, ngram + 1)]
        self.filename = "./data/" + character.replace(" ", "").replace(".", "").replace("&", "") + ".txt"
        self.character = character
        self.alpha = alpha
        self.smooth = smooth
        
        self.textContent = self.clean_input_document()
        self.textDict = list( set(word for line in self.textContent for word in line.split(" ") if word) )
        self.textDictSize = len(self.textDict)
        self.wordCounts = self.count_words(self.textContent)
        self.word2index = {w: i for i, w in enumerate(self.wordCounts[self.ngramCounts[0]].keys())}
        #self.probabilities = self.get_probability_matrix()
        
    def clean_input_document(self):
        if self.character == 'action':
            with open(self.filename) as f:
                contents = f.read().lower()
            translate = "?:!-\n"
            replace = "...  "
            delete = ",;_()\"\'[]"
            table = contents.maketrans(translate, replace, delete)
            contents = contents.translate(table).replace("mr. ", "mr ").replace("mrs. ", "mrs")
            text = contents.split(".")
            text = ["<s> " + t.strip() + " </s>" for t in text if t]
            return text
        
        with open(self.filename) as f:
            contents = f.read().lower()
        translate = "?:!-\n"
        replace = "...  "
        delete = ",;_()\""
        table = contents.maketrans(translate, replace, delete)
        contents = contents.translate(table).strip().replace("...", "").replace("mr. ", "mr ").replace("mrs. ", "mrs")
        contents = re.sub("''\[.*?\]''","", contents).replace("'", "")

        text = contents.split(".")
        text = ["<s> " + t.strip() + " </s>" for t in text if t]
        return text
            
    def count_words(self, contents):
        """Iterate through the contents and gather the counts of words"""
        wordCounts = {}
        for i in self.ngramCounts:
            if i == 0: # want the default to be the size of the corpus
                total = 0
                for line in contents:
                    words = line.split(" ")
                    words = [ w.strip() for w in words if w] #remove nulls
                    for word in words:
                        if word:
                            total += 1
                wordCounts[i] = defaultdict(lambda: total)
                continue
            else:
                counts = defaultdict(lambda: 0)
            for line in contents:
                words = line.split(" ")
                words = [ w.strip() for w in words if w] #remove nulls
                for k, word in enumerate(words): 
                    if k < (i-1) or not word:
                        continue
                    key = ""
                    for j in range(k-i+1, k+1):
                        key += words[j] + " "
                    counts[key.strip()] += 1
            wordCounts[i] = counts
        return wordCounts

    def model_smooth(self, x):
        return (self.wordCounts[self.ngramCounts[1]][x[0]] + self.alpha) / \
                    ( self.wordCounts[self.ngramCounts[0]][x[1]] + self.textDictSize*self.alpha )
    
    def model(self, x):
        return (self.wordCounts[self.ngramCounts[1]][x[0]] ) / \
                    ( self.wordCounts[self.ngramCounts[0]][x[1]]  )
    
    def get_probability_matrix(self):
 
        probabilities = []
        for wordA in self.wordCounts[self.ngramCounts[0]].keys():
            line = []
            for wordB in self.textDict:
                line.append(self.model([wordA + " " + wordB, wordA]))
            probabilities.append(line)
        return np.array(probabilities)
    
    def get_starting_word(self):
        potential = []
        for phrase in self.wordCounts[self.ngramCounts[0]].keys():
            if phrase[:3] == "<s>":
                potential.append(phrase)
        return random.choice(potential)

    def generate_sample_top_k(self, lm, index2word):
        """ Taken from - http://veredshwartz.blogspot.com/2019/08/text-generation.html
        Generates a string, sample a word from the top k probable words in the distribution at each time step.
        :param lm - the language model
        :param index2word - a mapping from the index of a word in the vocabulary to the word itself
        :param k - how many words to keep in the distribution """ 

        generated_sentence = self.get_starting_word()
        curr_token = None
        generated_tokens = 0

        while '</s>' not in generated_sentence and generated_tokens < self.maxTokens:
            #NEED TO CHOOSE A ROW ELEMENT -- LAST n-1 WORDS OF SENTENCE
            gen_list = generated_sentence.split()[-self.ngramCounts[0]:]
            gen_row = " ".join(gen_list)

            curr_distribution = lm(gen_row)  # vector of probabilities
            sorted_by_probability = np.argsort(curr_distribution) # sort by probability
            top_k = sorted_by_probability[-(self.k+1):] # keep the top k words
            
            selected_probs = [curr_distribution[t] for t in top_k]
            if selected_probs.count(selected_probs[0]) == len(selected_probs) : #all probabilities are indentical, randomly choose
                top_k = [np.random.choice(range(len(index2word)))]

            k_index2Word = [] #grab the top k words associated with top_k probabilities
            for index in top_k:
                k_index2Word.append(index2word[index])

            # normalize to make it a probability distribution again
            top_k = top_k / np.sum(top_k)

            selected_index = np.random.choice(range(len(k_index2Word)), p=top_k)
            curr_token = k_index2Word[int(selected_index)]
            generated_sentence += ' ' + curr_token   
            generated_tokens += 1

        return generated_sentence


    def generate_sample(self, lm, index2word):
        """ Taken from http://veredshwartz.blogspot.com/2019/08/text-generation.html
        Generates a string, sample a word from the distribution at each time step.
        :param lm - the language model
        :param index2word - a mapping from the index of a word in the vocabulary to the word itself """ 
        
        generated_sentence = self.get_starting_word()
        generated_tokens = 0
        curr_token = None

        while '</s>' not in generated_sentence and generated_tokens < self.maxTokens:
            #NEED TO CHOOSE A ROW ELEMENT -- LAST n-1 WORDS OF SENTENCE
            gen_list = generated_sentence.split()[-self.ngramCounts[0]:]
            gen_row = " ".join(gen_list)

            curr_distribution = lm(gen_row)  # vector of probabilities
            curr_distribution /= np.sum(curr_distribution)

            selected_index = np.random.choice(range(len(index2word)), p=curr_distribution)
            curr_token = index2word[int(selected_index)]
            generated_sentence += ' ' + curr_token
            generated_tokens += 1

        return generated_sentence
    
    def generate_sample_greedy(self, lm, index2word):
        generated_sentence = self.get_starting_word()
        generated_tokens = 0
        curr_token = None
        
        while '</s>' not in generated_sentence and generated_tokens < self.maxTokens:
            #NEED TO CHOOSE A ROW ELEMENT -- LAST n-1 WORDS OF SENTENCE
            gen_list = generated_sentence.split()[-self.ngramCounts[0]:]
            gen_row = " ".join(gen_list)
            
            curr_distribution = lm(gen_row)  # vector of probabilities
            curr_distribution /= np.sum(curr_distribution)

            selected_index = np.argmax(np.random.random(curr_distribution.shape) * \
                                       (curr_distribution==curr_distribution.max()) )
            curr_token = index2word[int(selected_index)]
            generated_sentence += ' ' + curr_token
            generated_tokens += 1

        return generated_sentence
    
    def lm(self, wordA):
        line = []
        for wordB in self.textDict:
            if self.smooth:
                line.append(self.model_smooth([wordA + " " + wordB, wordA]))
            else:
                line.append(self.model([wordA + " " + wordB, wordA]))
        return line

    def create_sentence(self, greedy=True, topk=False, maxTokens=25, k=2):
        self.k = k
        self.maxTokens = maxTokens
        #lm = lambda s: self.probabilities[self.word2index.get(s, -1), :]
        
        if greedy:
            sentence = self.generate_sample_greedy(self.lm, self.textDict)
        elif topk:
            sentence =  self.generate_sample_top_k(self.lm, self.textDict)
        else:
            sentence =  self.generate_sample(self.lm, self.textDict)
            
        return sentence.replace(" </s>", ".").replace("<s>", "")