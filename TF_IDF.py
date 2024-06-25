from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def transpose_and_sort(tfidf_df):
    tfidf_df_transposed = tfidf_df.T
    tfidf_df_transposed.columns = ['TF-IDF']
    tfidf_df_transposed = tfidf_df_transposed.sort_values(by='TF-IDF', ascending=False)
    return tfidf_df_transposed
  
def compute_tfidf(corpus, ngram_range):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    return tfidf_df, tfidf_vectorizer

def compute_tfidf_per_document(doc, ngram_range):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    return tfidf_df, tfidf_vectorizer
  
arm_text = " ".join(arm_corp)
comp_text = " ".join(comp_corp)

corpus = [arm_text, comp_text]

tfidf_df_words, tfidf_vectorizer_words = compute_tfidf(corpus, ngram_range=(1, 1))

tfidf_df_bigrams, tfidf_vectorizer_bigrams = compute_tfidf(corpus, ngram_range=(2, 2))

tfidf_df_words_transposed = tfidf_df_words.T
tfidf_df_words_transposed.columns = [f'Document_{i}' for i in range(1, tfidf_df_words_transposed.shape[1] + 1)]
tfidf_df_words_transposed = tfidf_df_words_transposed.apply(lambda x: x.sort_values(ascending=False).values)

tfidf_df_bigrams_transposed = tfidf_df_bigrams.T
tfidf_df_bigrams_transposed.columns = [f'Document_{i}' for i in range(1, tfidf_df_bigrams_transposed.shape[1] + 1)]
tfidf_df_bigrams_transposed = tfidf_df_bigrams_transposed.apply(lambda x: x.sort_values(ascending=False).values)

print("TF-IDF для слов (униграмм):")
print(tfidf_df_words_transposed)

print("\nTF-IDF для биграмм:")
print(tfidf_df_bigrams_transposed)

tfidf_df_words_doc1, tfidf_vectorizer_words_doc1 = compute_tfidf_per_document(arm_text, ngram_range=(1, 1))
tfidf_df_words_doc2, tfidf_vectorizer_words_doc2 = compute_tfidf_per_document(comp_text, ngram_range=(1, 1))

tfidf_df_bigrams_doc1, tfidf_vectorizer_bigrams_doc1 = compute_tfidf_per_document(arm_text, ngram_range=(2, 2))
tfidf_df_bigrams_doc2, tfidf_vectorizer_bigrams_doc2 = compute_tfidf_per_document(comp_text, ngram_range=(2, 2))

tfidf_df_words_doc1_transposed = transpose_and_sort(tfidf_df_words_doc1)
tfidf_df_words_doc2_transposed = transpose_and_sort(tfidf_df_words_doc2)

tfidf_df_bigrams_doc1_transposed = transpose_and_sort(tfidf_df_bigrams_doc1)
tfidf_df_bigrams_doc2_transposed = transpose_and_sort(tfidf_df_bigrams_doc2)

print("\nTF-IDF для слов (униграмм) Document 1:")
print(tfidf_df_words_doc1_transposed)

print("\nTF-IDF для слов (униграмм) Document 2:")
print(tfidf_df_bigrams_doc2_transposed)

print(tfidf_df_words_doc2_transposed)

print("\nTF-IDF для биграмм Document 1:")
print(tfidf_df_bigrams_doc1_transposed)

print("\nTF-IDF для биграмм Document 2:")
print(tfidf_df_bigrams_doc2_transposed)
