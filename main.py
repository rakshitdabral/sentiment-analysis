import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
import re
import os


input_file = pd.read_excel('input.xlsx')
input_file.head(10)

input_file.tail(3)

len(input_file)

for index, row in input_file.iterrows():
    url = row['URL']
    url_id = row['URL_ID']
    # print(index)

    article = " "  # storing the extracted data

    # code to fetch data
    try:
        page = requests.get(url)
        # print(page)
    except:
        print("Cant get response")

    try:
        soup = BeautifulSoup(page.content, 'html.parser')
        title = soup.find('h1').get_text()
        for p in soup.find_all('p'):
            article += p.get_text()
        # print(article)
        file_name = 'extracted/' + str(url_id) + '.txt'
        with open(file_name, 'w') as file:
            file.write(title + '\n' + article)
        print(f"Success {url_id}")
    except:
        print(f"not found {url_id}")


input_file.drop([35,48,18,24,25,78,87], axis = 0, inplace=True)

text_dir = 'extracted/'
stop_dir = 'StopWords/'
master_dir = 'MasterDictionary/'

    #setting up the stopwords
stop_words = set()
for file in os.listdir(stop_dir):
      with open(os.path.join(stop_dir,file),'r') as f:
            stop_words.update(set(f.read().splitlines()))
stop_words

#setting up positive words
pos_words = set()
neg_words = set()

for files in os.listdir(master_dir):
  if files =='positive-words.txt':
    with open(os.path.join(master_dir,files),'r') as f:
      pos_words.update(f.read().splitlines())
  else:
    with open(os.path.join(master_dir,files),'r') as f:
      neg_words.update(f.read().splitlines())

pos_words
neg_words

# tokenizing text file and creating a doc for it

docs = []
for text_file in os.listdir(text_dir):
  with open(os.path.join(text_dir,text_file),'r') as f:
    text = f.read()
    words = word_tokenize(text)
    broken = [word for word in words if word.lower() not in stop_words]
    docs.append(broken)
docs

len(docs)
docs[3] # checking if it works


positive_words = []
negative_words =[]
positive_score = []
negative_score = []
polarity_score = []
subjectivity_score = []

for i in range(len(docs)):
    for word in docs[i]:
        if word.lower() in pos_words:
            positive_words.append(word)
            # print(positive_words)
    for word in docs[i]:
        if word.lower() in neg_words:
            negative_words.append(word)

    positive_score.append(len(positive_words[i]))
    negative_score.append(len(negative_words[i]))
    polarity_score.append(
        (positive_score[i] - negative_score[i]) / ((positive_score[i] + negative_score[i]) + 0.000001))
    subjectivity_score.append((positive_score[i] + negative_score[i]) / ((len(docs[i])) + 0.000001))


len(positive_score)
len(negative_score)
len(polarity_score)
len(subjectivity_score)

average_sentence_length = []
percentage_of_complex_words = []
fog_index = []
number_of_complex_words = []
syllable_count_per_word =[]
average_number_of_words_per_sentence = []
stp_words = set(stopwords.words('english'))


def finding_sentences(file):
    with open(os.path.join(text_dir, file), 'r') as f:
        text = f.read()

        total_sentences = []
        total_words = []
        sentences_broken = sent_tokenize(text)
        for x in sentences_broken:
            total_sentences.append(sentences_broken)
            for word in x.split():
                total_words.append(word)

        total_sentences_length = len(total_sentences)
        total_word_length = len(total_words)

        # print(toal_word_length)

        pattern = r'[^\w\s.]'
        text = re.sub(pattern, ' ', text)
        sentences = re.split(r'[.!?]\s*', text)

        num_sentences = len(sentences)

        word_in_sentences = []
        for word in text.split():
            if word.lower() not in stp_words:
                word_in_sentences.append(word)
        num_of_words = len(word_in_sentences)

        complex_word = []
        for word in word_in_sentences:

            vowels = 'aeiou'
            syllables = 0
            for letter in word:
                if letter.lower() in vowels:
                    syllables = syllables + 1

            if syllables > 2:
                complex_word.append(word)

        syllable_count = 0
        syllable_count_word = []

        for word in word_in_sentences:
            if word.endswith('es'):
                word = word[:-2]
            elif word.endswith('ed'):
                word = word[:-2]
            vowels = 'aeiou'
            for letter in word:
                if letter.lower() in vowels:
                    syllable_count = syllable_count + 1
            if syllable_count >= 1:
                syllable_count_word.append(word)

        number_of_complex_words = len(complex_word)

        avg_sen_len = num_of_words / num_sentences
        if num_of_words == 0:
            per_complex_word = 0
        else:
            per_complex_word = len(complex_word) / num_of_words

        fog = 0.4 * (avg_sen_len + per_complex_word)

        if syllable_count == 0:
            syllable_count_per_word = 0
        else:
            syllable_count_per_word = syllable_count / len(syllable_count_word)

        if total_word_length == 0:
            average_number_of_words_per_sentence = 0
        else:
            average_number_of_words_per_sentence = total_word_length / total_sentences_length

        return avg_sen_len, per_complex_word, fog, syllable_count_per_word, average_number_of_words_per_sentence, number_of_complex_words


for file in os.listdir(text_dir):
    x,y,z,k,l, m= finding_sentences(file)
    average_sentence_length.append(x)
    percentage_of_complex_words.append(y)
    fog_index.append(z)
    syllable_count_per_word.append(k)
    average_number_of_words_per_sentence.append(l)
    number_of_complex_words.append(m)

len(average_number_of_words_per_sentence)

len(percentage_of_complex_words)
len(number_of_complex_words)
len(fog_index)
len(syllable_count_per_word)
len(average_number_of_words_per_sentence)

personal_pronoun = [ ]
def personal_pronouns(file):
  with open(os.path.join(text_dir,file), 'r') as f:
    text = f.read()
    pronoun_regex = r"\b(I|me|my|mine|you|your|yours|he|him|his|she|her|hers|it|its|they|them|their|theirs|we|us|our|ours)\b"
    pronoun_regex = rf"{pronoun_regex}(?!\sUS)"
    matches = re.findall(pronoun_regex, text, flags=re.IGNORECASE)
    count = len(matches)
  return count

for file in os.listdir(text_dir):
  x = personal_pronouns(file)
  personal_pronoun.append(x)
len(personal_pronoun)

avg_word_len = []
word_count = [ ]


def avg_word(file):
    with open(os.path.join(text_dir, file), 'r') as f:
        text = f.read()
        pattern = r'[^\w\s.]'
        text = re.sub(pattern, ' ', text)
        sentences = re.split(r'[.!?]\s*', text)
        num_sentences = len(sentences)

        word_in_sentences = []
        for word in text.split():
            if word.lower() not in stp_words:
                word_in_sentences.append(word)
        num_of_words = len(word_in_sentences)

        length = 0
        for word in word_in_sentences:
            length = length + len(word)

        if num_of_words == 0:
            average_word_length = 0
        else:
            average_word_length = length / num_of_words
        return num_of_words, average_word_length


for file in os.listdir(text_dir):
  x, y = avg_word(file)
  word_count.append(x)
  avg_word_len.append(y)

len(avg_word_len)
len(word_count)

output_data = pd.read_excel('Output Data Structure.xlsx')
output_data

# output_data.drop([35,48], axis = 0, inplace=True)

variables_present = [positive_score,
            negative_score,
            polarity_score,
            subjectivity_score,
            average_sentence_length,
            percentage_of_complex_words,
            fog_index,
            average_number_of_words_per_sentence,
            number_of_complex_words,
            word_count ,
            syllable_count_per_word,
            personal_pronoun,
            avg_word_len]

# variable = []

# for x in output_data.columns:
#     variable.append(x)

# df = pd.DataFrame(columns=variable)

# data = {'POSITIVE SCORE': positive_score ,
#         'NEGATIVE SCORE' : negative_score,
#         'POLARITY SCORE' : polarity_score,
#        'SUBJECTIVITY SCORE': subjectivity_score,
#        'AVG SENTENCE LENGTH' : average_sentence_length,
#        'PERCENTAGE OF COMPLEX WORDS': percentage_of_complex_words,
#         'FOG INDEX' : fog_index,
#         'AVG NUMBER OF WORDS PER SENTENCE' : average_number_of_words_per_sentence,
#         'COMPLEX WORD COUNT' : number_of_complex_words,
#         'WORD COUNT' : word_count,
#         'SYLLABLE PER WORD' : syllable_count_per_word,
#         'PERSONAL PRONOUNS' : personal_pronoun,
#         'AVG WORD LENGTH' : avg_word_len
#        }

# df = pd.DataFrame(data)
# df.drop(
# df

output_data

for i, var in enumerate(variables_present):
  output_data.iloc[:,i+2] = var

output_data.to_csv('output_data.csv')





