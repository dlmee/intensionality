import json
import nltk
from nltk.corpus import wordnet as wn


def count_senses(word_list):
    sense_counts = {}
    for word, freq in word_list:
        synsets = wn.synsets(word, pos=wn.ADJ)  # Filter for noun part-of-speech
        count = len(synsets)
        if count not in sense_counts:
            sense_counts[count] = []
        sense_counts[count].append((word, freq))# Store count or 0 if no synsets found
    
    return sense_counts

def is_not_bidirectional_constituent(test_string, string_list):
    if test_string[-3:] == 'est':
        return False
    if test_string[-2:] == 'er':
        return False
    # Iterate through each string in the list
    for s in string_list:
        # Check if the test string is a substring of the current string or vice versa
        if test_string in s or s in test_string:
            return False  # Return False if a bidirectional substring relationship is found
    return True 


if __name__ == "__main__":

    with open("first_pass.json", "r") as inj:
        mydata = json.load(inj)

    for k,v in mydata.items():
        #print(k)
        v = list(v.items())
        v = sorted(v, key= lambda x:x[1], reverse=True)
        if k == 'ADJ':
            myoutput = [(word, count) for word, count in v if not word[0].isupper()]
            myoutput = count_senses(myoutput)
            #myoutput = sorted(list(myoutput.items()), key = lambda x:x[1], reverse=True)
            targets = ['alleged', 'former', 'future', 'hypothetica', 'impossible', 'likely', 'mere', 'mock', 'loose', 'wide', 'white', 'naive', 'severe', 'hard', 'intelligent', 'ripe', 'necessary', 'past', 'possible', 'potential', 'presumed', 'probable', 'putative', 'theoretica', 'modern', 'black', 'free', 'safe', 'vile', 'nasty', 'meagre', 'stable']
            final = [(triad[0], k, triad[1]) for k,v in myoutput.items() for triad in v if triad[0] in targets]
            used = targets
            while len(final) < 100:
                for k,v in myoutput.items():
                    if k == 0: continue
                    if v:
                        word, freq = v.pop(0)
                        if is_not_bidirectional_constituent(word, used):
                            final.append((word, k, freq))
                            used.append(word)
                    if len(final) == 100: break
            final = sorted(final, key= lambda x:x[1], reverse=True)
            final = [word for word, count, freq in final]


    with open("my_adj_candidates.json", "w") as outj:
        json.dump(final, outj, indent=4, ensure_ascii=False)