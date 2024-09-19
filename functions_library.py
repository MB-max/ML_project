import numpy as np
import pandas as pd

# Processing Data functions

from nltk.corpus import stopwords 
ENGLISH_STOP_WORDS = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def spl_tokenizer(sentence):

    # dividi la frase in parole
    listofwords = sentence.split(' ')
    listoflemmatized_words = []
        
    # Remove stopwords e ogni token con stringhe vuote
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Lemmatize words
            token = WordNetLemmatizer().lemmatize(word)            
            
            #add r_ prefisso per indicare una parola del campo recensione
            try:
                if tfidf.type == 'review':
                    token = 'r_' + token
            except:
                pass

            #append
            listoflemmatized_words.append(token)

    return listoflemmatized_words



def sps_tokenizer(sentence):

    # dividi la frase in parole
    listofwords = sentence.split(' ')
    listofstemmed_words = []
    #instantiate stemmer
    stemmer = PorterStemmer() 

    # Remove stopwords e ogni token con stringhe vuote
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Lemmatize words
            token = stemmer.stem(word)  

            #add r_ prefisso per indicare una parola del campo recensione
            try:
                if tfidf.type == 'review':
                    token = 'r_' + token
            except:
                continue

            #append
            listofstemmed_words.append(token)

    return listofstemmed_words




def tfidf(dataframe_column, tokenizer, min_df=0.02, max_df=0.8, ngram_range=(1,1)):

    #0. Per la tokenizzazione, è necessario determinare quale prefisso aggiungere a ciascun token
    #prendi il nome della colonna da cui vengono generati i tokend
    column_name = dataframe_column.name
    #assegnare un attributo (chiamato tipo) alla funzione tfidf per indicare se i token provengono dalla revisione
    if column_name == 'clean_reviews':
        tfidf.type = 'review'
    else:
        tfidf.type = 'none'

    # 1. Inizializza
    vectorizer = TfidfVectorizer(min_df = min_df, max_df = max_df, tokenizer = tokenizer, ngram_range = ngram_range)
    
    # 2. Adatta 
    vectorizer.fit(dataframe_column)
    
    # 3. Transform
    reviews_tokenized = vectorizer.transform(dataframe_column)
    
    # Estrai info e inseriscile in df
    tokens = pd.DataFrame(columns=vectorizer.get_feature_names_out(), data=reviews_tokenized.toarray())
    
    return tokens

