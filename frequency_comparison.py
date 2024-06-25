import docx
import re
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from collections import Counter
import pandas as pd

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
mystem = Mystem()

def read_word_file(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def preprocess_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text, language='russian')
    tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = mystem.lemmatize(' '.join(tokens))
    lemmatized_tokens = [token for token in lemmatized_tokens if token.strip()]
    return lemmatized_tokens

file_path_1 = "АЛЬТЕРНАТИВНЫЙ КОРПУС.docx"  
file_path_2 = "Армянский корпус последний июнь 2024.docx"

text1 = read_word_file(file_path_1)
text2 = read_word_file(file_path_2)

comp_corp = preprocess_text(text1)
arm_corp = preprocess_text(text2)

def create_frequency_dict(tokens):
    return Counter(tokens)

def compare_frequencies(freq_dict1, freq_dict2):
    unique_to_corpus1 = {}
    unique_to_corpus2 = {}
    
    for word, freq in freq_dict1.items():
        if word not in freq_dict2:
            unique_to_corpus1[word] = freq
        elif freq > freq_dict2[word]:
            unique_to_corpus1[word] = freq - freq_dict2[word]
    
    for word, freq in freq_dict2.items():
        if word not in freq_dict1:
            unique_to_corpus2[word] = freq
        elif freq > freq_dict1[word]:
            unique_to_corpus2[word] = freq - freq_dict1[word]
    
    return unique_to_corpus1, unique_to_corpus2

freq_dict_comp = create_frequency_dict(comp_corp)
freq_dict_arm = create_frequency_dict(arm_corp)

unique_to_comp_corp, unique_to_arm_corp = compare_frequencies(freq_dict_comp, freq_dict_arm)

def freq_dict_to_sorted_df(freq_dict):
    df = pd.DataFrame(freq_dict.items(), columns=['Word', 'Frequency'])
    df = df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return df

unique_to_comp_corp, unique_to_arm_corp = freq_dict_to_sorted_df(unique_to_comp_corp), freq_dict_to_sorted_df(unique_to_arm_corp)

print(unique_to_comp_corp, unique_to_arm_corp)
