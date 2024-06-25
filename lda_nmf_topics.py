from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from tqdm import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning

def prepare_corpus(texts):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
    dtm = vectorizer.fit_transform(texts)
    return dtm, vectorizer

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(" ".join(topic_words))
    return topics
  
def fit_lda_with_progress(dtm, n_topics, max_iter):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=1,  
        random_state=42,
        learning_method='online',
        learning_decay=0.7,
        learning_offset=50.0
    )
    for i in tqdm(range(max_iter), desc="Fitting LDA"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            lda.partial_fit(dtm)
    return lda
  
def fit_nmf(dtm, n_topics, max_iter):
    nmf = NMF(
        n_components=n_topics,
        max_iter=max_iter,  
        random_state=42,
        init='nndsvda',
        alpha_W=0.1,
        alpha_H=0.1
    )
    nmf.fit(dtm)
    return nmf
  
stop_words = stopwords.words('russian')

arm_texts = ["".join(tokens) for tokens in arm_corp]
comp_texts = ["".join(tokens) for tokens in comp_corp]

dtm_arm, vectorizer_arm = prepare_corpus(arm_texts)
dtm_comp, vectorizer_comp = prepare_corpus(comp_texts)

n_topics = 10
n_top_words = 10
max_iter = 50

lda_arm = fit_lda_with_progress(dtm_arm, n_topics, max_iter)
lda_comp = fit_lda_with_progress(dtm_comp, n_topics, max_iter)

topics_arm_lda = display_topics(lda_arm, vectorizer_arm.get_feature_names_out(), n_top_words)
topics_comp_lda = display_topics(lda_comp, vectorizer_comp.get_feature_names_out(), n_top_words)

nmf_arm = fit_nmf(dtm_arm, n_topics, max_iter)
nmf_comp = fit_nmf(dtm_comp, n_topics, max_iter)

topics_arm_nmf = display_topics(nmf_arm, vectorizer_arm.get_feature_names_out(), n_top_words)
topics_comp_nmf = display_topics(nmf_comp, vectorizer_comp.get_feature_names_out(), n_top_words)

print(topics_arm_lda)
print(topics_comp_lda)
print(topics_arm_nmf)
print(topics_comp_nmf)
