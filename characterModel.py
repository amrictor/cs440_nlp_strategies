from collections import defaultdict, Counter
import numpy as np
import random
import bisect
import re

class character_model:
    def __init__(self, filename="characterInteractions.txt", numberLines=20):
        self.characters = []
        self.content = []
        self.numberLines = numberLines
        self.create_character_model(filename)
        self.generate_character_order()
    
    def create_character_model(self, filename):  
        #READ IN DATA AND CREATE TWO LISTS: CHARACTERS AND THEIR CHANCE OF BEING SELECTED
        with open(filename) as f:
            char_probs = {}
            for line in f:
                character, followers = line.split("\t")
                followers = followers.strip().split(" ")
                charList, probList = [], []
                for follow in followers:
                    char, occurance = follow.split(";")
                    charList.append(char)
                    probList.append(int(occurance) )
                total = sum(probList)
                probList = [round(p / total, 4) for p in probList]
                char_probs[character] = (charList, probList)
                
        #SAVE ALL THE CHARACTERS
        self.characters = list( char_probs.keys() )

        #FINALLY, GENERATE A WEIGHTED SAMPLER FROM THE CHAR LIST AND PROB LIST FOR EACH CHARACTER
        character_model = {}
        for char, lists in char_probs.items():
            character_model[char] = self.weighted_sampler(lists[0], lists[1])
        self.character_model = character_model
        
    def weighted_sampler(self, charList, probList):
        totals = []
        for p in probList:
            totals.append(p + totals[-1] if totals else p)
        return lambda: charList[bisect.bisect(totals, random.uniform(0, totals[-1]))]

    #SELECT A CHARACTER TO FOLLOW THE GIVEN CHARACTER X
    def next_character(self, x):
        return self.character_model[x]()
    
    def generate_character_order(self):
        #RANDOMLY CHOOSE FIRST CHARACTER 
        first = random.choice(self.characters)
        order = [first]
        
        for i in range(self.numberLines - 1):
            order.append(self.next_character(order[i]))
        self.ordering = order
    
    def get_ordering(self):
        return self.ordering