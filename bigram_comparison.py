import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def get_bigrams(tokens):
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    return sorted(bigram for bigram, score in scored)

bigrams_arm = get_bigrams(arm_corp)
bigrams_comp = get_bigrams(comp_corp)

def bigram_frequency(bigrams):
    freq = Counter(bigrams)
    return freq

freq_bigrams_arm = bigram_frequency(bigrams_arm)
freq_bigrams_comp = bigram_frequency(bigrams_comp)

def compare_bigrams(freq1, freq2):
    unique_to_corp1 = {bigram: freq1[bigram] for bigram in freq1 if bigram not in freq2}
    unique_to_corp2 = {bigram: freq2[bigram] for bigram in freq2 if bigram not in freq1}
    common_bigrams = {bigram: (freq1[bigram], freq2[bigram]) for bigram in freq1 if bigram in freq2}
    return unique_to_corp1, unique_to_corp2, common_bigrams

unique_bigrams_arm, unique_bigrams_comp, common_bigrams = compare_bigrams(freq_bigrams_arm, freq_bigrams_comp)

unique_bigrams_arm = freq_dict_to_sorted_df(unique_bigrams_arm)
unique_bigrams_comp = freq_dict_to_sorted_df(unique_bigrams_comp)
common_bigrams = freq_dict_to_sorted_df(common_bigrams)

print(unique_bigrams_arm)
print(unique_bigrams_comp)
print(common_bigrams)
