import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import string
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import gc
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier, to_graphviz

nltk.download('vader_lexicon')

df = pd.read_csv('Preprocessed_McDonald_s_Reviews.csv')
df.head()

df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()
df.dropna(inplace=True)
df.info()

# Crea TF-IDF vectorizer
vectorizer = TfidfVectorizer()
df['clean_reviews'].fillna('', inplace=True)

# Adatta il vettorizzatore ai documenti e trasforma i documenti in matrice TF-IDF
tfidf_matrix = vectorizer.fit_transform(df['clean_reviews'])

# feature names
feature_names = vectorizer.get_feature_names_out() 

# Inizializza il sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Calcula polarita per ogni text del dataset
polarities = []
for text in df['clean_reviews']:
    sentiment = sia.polarity_scores(text)
    polarity = sentiment['compound']  # compound score ranges from -1 to 1
    
    polarities.append(polarity)

# Calcula polarita totale
dataset_polarity = sum(polarities) / len(polarities)

print(f"Polarita totale: {dataset_polarity}")

df['Polarity']=polarities
df.head(5)

# Funzione per pre processare il campo review text
def preprocess_text(text):
    #Converti il ​​testo in minuscolo e dividilo in parole
    words = text.lower().split()
    
    #Rimuovi punteggiatura e numeri
    words = [word.translate(str.maketrans('', '', string.punctuation + string.digits)) for word in words]
    
    # Remove stop
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stopwords]
    
    #Usa uno stemmer Porter
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
        
    # Return lista di parole
    return ' '.join(words)
df['clean_reviews'] = df['clean_reviews'].apply(preprocess_text)

#Addestrare il modello Word2Vec sul review text
sentences = [review.split() for review in df['clean_reviews']]
model = gensim.models.Word2Vec(sentences, vector_size=100, min_count=1, workers=4)


def get_word2vec_embedding(text):
    # Tokenize il text in parole
    words = word_tokenize(text)
    
    # Addestrare il modello Word2Vec sulle parole tokenized
    model = Word2Vec([words], min_count=1, size=100)
    
    #Aggrega
    word_vectors = model.wv
    
    return word_vectors

print(gc.collect())

doc_embeddings = []
for sentence in df['clean_reviews']:
    words = sentence.split()
    vectors = [model.wv.get_vector(word) for word in words if word in model.wv.key_to_index]
    if vectors:
        mean_vector = np.mean(vectors, axis=0)
        doc_embeddings.append(mean_vector)
    else:
        doc_embeddings.append(np.zeros(model.vector_size))
