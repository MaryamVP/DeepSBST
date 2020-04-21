import pandas as pd
import re,random
from nltk.corpus import wordnet
import wordninja
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def divide_name(mathod_string):
    match_ret = re.search('\w+\s*\(',mathod_string)
    
    if match_ret:
        method_name = match_ret.group()[:-1].strip()
        print(method_name)
        
        two_word = re.search(r'[A-Z]', method_name)
        print(two_word)
        
        if two_word:
            idx = re.search(r'[A-Z]', method_name).start()
            print(idx)
            if idx > 0:
                first_word = method_name[:idx]
                second_word = method_name[idx:]

                return first_word.strip(), second_word.strip()
            else:
                return method_name.strip(), method_name.strip()

        else:
            return method_name.strip(), method_name.strip()

        
def word_synonym_replacement(word):
    if len(word) <=3:
        return word + '_new'
    word_set = wordninja.split(word)
    while True:
        if word_set == []:
            return word + '_new'
        word_tar = random.choice(word_set)
        word_syn = wordnet.synsets(word_tar)
        if word_syn == []:
            word_set.remove(word_tar)
        else:
            break
    word_ret = []
    for syn in word_syn:
        word_ret = word_ret + syn.lemma_names()
        if word_tar in word_ret:
            word_ret.remove(word_tar)
    print(word_ret)
    try:
        word_new = random.choice(word_ret)
    except:
        word_new = word

    return word.replace(word_tar,word_new)

def words_to_replace(method_string):
    
    first_word, second_word = divide_name(method_string)
    print(first_word, second_word)
    
    if first_word != second_word:
        new_first = method_string.replace(method_string, word_synonym_replacement(first_word))
        new_sec = method_string.replace(method_string, word_synonym_replacement(second_word))
    
        return [first_word, new_first.lower(), second_word , new_sec.lower()]
    else:
        return [second_word, method_string.replace(method_string, word_synonym_replacement(second_word))]

def replace_token(method_string, i):
    t = []

    names = words_to_replace(method_string)
    
    tokens = pd.read_csv('train.tokens.txt', sep="\n", header=None)
    
    if len(names) > 2:
            token = tokens[0][i]
            print(token)
            token = token.replace(names[0], names[1])
            token = token.replace(names[2], names[3])

            token = token.replace('_',' ')
            token = token.replace('-',' ')
            t.append(token)

    else:
            token = tokens[0][i]
            token = token.replace(names[0], names[1])

            token = token.replace('_',' ')
            token = token.replace('-',' ')
            t.append(token)



    final_tokens = pd.DataFrame(t)
    final_tokens.to_csv('new_token.txt', index=False)
    
    return t
    
def replace_methname(method_string, i):
    m = []
    
    names = words_to_replace(method_string)
    
    methname = pd.read_csv('train.methname.txt', sep="\n", header=None)
    
    if len(names) > 2:
            mth = methname[0][i]
            mth = mth.replace(names[0], names[1])
            mth = mth.replace(names[2], names[3])
            mth = mth.replace('_',' ')
            mth = mth.replace('-',' ')
            m.append(mth)

    else:
            mth = methname[0][i]
            mth = mth.replace(names[0], names[1])
            mth = mth.replace('_',' ')
            mth = mth.replace('-',' ')
            m.append(mth)



    final_methodname = pd.DataFrame(m)
    final_methodname.to_csv('new_methname.txt', index=False)
    
    return m

def text_to_array(data):
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    max_len = max([len(s.split()) for s in data])
    print('Max len is:', max_len)
    
    vocab_size = len(tokenizer.word_index) + 1

    input_data = tokenizer.texts_to_sequences(data)

    input_data_pad = pad_sequences(input_data, maxlen = max_len, padding='post')

    
    return input_data_pad 
    