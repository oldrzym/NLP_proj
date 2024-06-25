import re
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from nltk.tokenize import sent_tokenize
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))
mystem = Mystem()

def preprocess_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    
    text = text.lower()
    
    sentences = sent_tokenize(text, language='russian')
    
    processed_sentences = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence, language='russian')
        
        tokens = [token for token in tokens if token not in stop_words]
        
        lemmatized_tokens = mystem.lemmatize(' '.join(tokens))
        
        lemmatized_tokens = [token for token in lemmatized_tokens if token.strip()]
        
        processed_sentences.append(lemmatized_tokens)
    
    return processed_sentences

word2vec_arm = preprocess_text(text2)
word2vec_comp = preprocess_text(text1)

def train_word2vec(corpus, vector_size=100, window=5, min_count=2, workers=4):
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

word2vec_arm = train_word2vec(word2vec_arm)
word2vec_comp = train_word2vec(word2vec_comp)

words_arm = list(word2vec_arm.wv.index_to_key)
words_comp = list(word2vec_comp.wv.index_to_key)

vectors_arm = [word2vec_arm.wv[word] for word in words_arm]
vectors_comp = [word2vec_comp.wv[word] for word in words_comp]

all_vectors = vectors_arm + vectors_comp
all_words = words_arm + words_comp

pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_vectors)

plt.figure(figsize=(14, 7))

plt.scatter(pca_result[:len(vectors_arm), 0], pca_result[:len(vectors_arm), 1], c='red', label='Armenian Corpus')

plt.scatter(pca_result[len(vectors_arm):, 0], pca_result[len(vectors_arm):, 1], c='blue', label='Comparative Corpus')

plt.legend()
plt.title('PCA of Word Embeddings for Armenian and Comparative Corpora')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
