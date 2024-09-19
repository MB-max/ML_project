import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA
import joblib
from pyLDAvis import sklearn as sklearn_lda
import pyLDAvis
import pickle 
import os
import functions_library as fl
from nltk.corpus import stopwords 
ENGLISH_STOP_WORDS = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('Preprocessed_McDonald_s_Reviews.csv')

df = df.astype({'clean_reviews':'string'})
#clean_review ha 145 valori null da rimuovere
print(df.isna().sum())
df = df.dropna()

#review tokens preceduti da 'r_'
review_tokens = fl.tfidf(df['clean_reviews'], tokenizer=fl.spl_tokenizer, ngram_range=(1,2), min_df=1000)

#caratteristiche numeriche (no tokens)
df2=df.select_dtypes(include=['int32','int64'])
#inizializza minmaxscaler
mm = MinMaxScaler()
#trasforma i dati usando minmaxscalar
df3 = mm.fit_transform(df2)
#crea nuovo df con questi dati
df_scaled = pd.DataFrame(df3, columns = df2.columns)
#salva
del df3, df2

#concatena 
df_final = pd.concat([df_scaled, review_tokens], axis = 1)


#save data
df_final.to_csv('processed_McDonald_s_Reviews.csv')
print(df_final.tail())
#salva in memoria
del review_tokens, df_scaled, df

# Create a PCA instance: pca
pca = PCA(n_components=64)
pcs = pca.fit_transform(df_final)

# Graico di varianza
plt.figure(figsize=(10,4))
plt.plot(range(64), pca.explained_variance_ratio_[0:64])
plt.xlabel('PCA features')
plt.ylabel('explained variance %')
plt.xticks(range(0,65,25))
plt.title('Explained variance % of Principal Components')
plt.show()

# Grafico comulativo di varianza
plt.figure(figsize=(10,4))
plt.plot(range(64), pca.explained_variance_ratio_.cumsum()[0:64])
plt.xlabel('PCA features')
plt.ylabel('cumulative sum of explained variance %')
plt.xticks(range(0,65,25))
plt.title('Cumulative Sum of Explained variance % of Principal Components')
plt.show()

#Varianza dei primi 30 PCs
pca.explained_variance_ratio_.cumsum()[30]

#Varianza dei primi 45 PCs
pca.explained_variance_ratio_.cumsum()[45]

#Varianza dei primi 63 PCs
pca.explained_variance_ratio_.cumsum()[63]
# Save componenti
PCA_components = pd.DataFrame(pcs)

#Grafico
plt.figure()
plt.scatter(PCA_components[0],PCA_components[1], alpha=.05, color='blue')
plt.xlabel('PCA 0')
plt.ylabel('PCA 1')
plt.show()

#iniziamo cercando di determinare il numero ottimale di cluster osservando i punteggi di inerzia, 
# passando attraverso diversi valori K e aggiungendo il punteggio di inerzia all'elenco
k_values = range(2,12)
inertia_scores = []

for k in k_values:
    #inizializza
    kmeans_model = KMeans(n_clusters=k, verbose=1)
    #Adatta ai primi 63 Pcs
    kmeans_model.fit(PCA_components.iloc[:,0:63])
    #genera lo score
    inertia = kmeans_model.inertia_
    #aggingilo agli altri score
    inertia_scores.append(inertia)
    
plt.figure()
plt.plot(k_values, inertia_scores)
plt.xlabel('num of clusters')
plt.ylabel('score')
plt.title('inertia')
plt.xticks(k_values)
plt.grid()
plt.show()

#KMeans con 10 clusters
kmeans_model10 = KMeans(n_clusters=10, verbose=1)
#Adatta KMeans su 63 PCs
kmeans_model10.fit(PCA_components.iloc[:,0:63])

plt.figure()
plt.scatter(PCA_components[0],PCA_components[1], c=kmeans_model10.labels_)
plt.show()

#Salva il modello per dopo
joblib.dump(kmeans_model10, 'kmeans_model10.pkl')

#Inizializza 2 matrici
# 1) data con principal components (PCs)
# 2) PCs e features
data_pc_matrix = PCA_components.iloc[:,0:63].to_numpy()
pc_feature_matrix = pca.components_[0:63,:]
#si effettuano moltiplicazioni tra matrici
df_recon = pd.DataFrame(np.matmul(data_pc_matrix,pc_feature_matrix), columns=df_final.columns)

#add cluster labels al dataframe
df_recon['cluster'] = kmeans_model10.labels_

#saving the df to the computer in a compressed h5 format (saves faster than other compression techniques)
df_recon.to_hdf('df_recon_10_kmeans.h5', key='df', mode='w')

######################## LDA #########################

df4 = pd.read_csv('Preprocessed_McDonald_s_Reviews.csv')
df4 = df4.astype({'clean_reviews':'string'})
#clean_review ha 145 valori null da rimuovere
print(df4.isna().sum())
df4 = df4.dropna()
review_text = df4['clean_reviews']

review_text
#utilizzare le stesse impostazioni utilizzate per il clustering KMeans per essere coerenti
vectorizer = TfidfVectorizer(min_df = 1000, tokenizer = fl.spl_tokenizer, ngram_range = (1,2))

#prendi tokens da clean_reviews
word_matrix = vectorizer.fit_transform(review_text)

def print_topics(model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(",".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
#Impostazione del numero di argomenti e anche del numero massimo di parole che vogliamo vedere dal modello
number_topics = 10
number_words = 10

# Crea e adatta LDA model
lda = LDA(n_components=number_topics, n_jobs=4, verbose=1)
lda.fit(word_matrix)

# stampa i topic trovati nel LDA model
print("Topics found via LDA:")
print_topics(lda, vectorizer, number_words)

#saving model to computer
joblib.dump(lda,'lda_10.pkl')
lda = joblib.load('lda_10.pkl')

#creating file path
LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(number_topics))
LDAvis_data_filepath

# preparing the LDA model to be saved in the LDAvis visualizer
LDAvis_prepared = sklearn_lda.prepare(lda, word_matrix, vectorizer)

#saving LDA model into LDAvis visualization
with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(LDAvis_prepared, f)
    
#saving LDA model into LDAvis visualization html file
pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(number_topics) +'.html')

###################### K MEANS CLUSTER ###########################

#leggere nel dataframe salvato dalla modellazione KMeans che contiene le etichette dei cluster
df = pd.read_hdf('df_recon_10_kmeans.h5')

print(df.head())
#prendi i nomi dei non-token features and cluster per analisi
non_tokens_clust = list(df.columns[0:12]) + ['cluster']

#prendi token features and cluster per analisi
tokens_clust = list(df.columns[12:])

#prendi average rating per cluster
print(df.loc[:,non_tokens_clust].groupby('cluster').mean().iloc[:,0].sort_values(ascending=False))

#prendi average rating2 per cluster
print(df.loc[:,non_tokens_clust].groupby('cluster').mean().iloc[:,2].sort_values(ascending=False))

#prendi i valori principali dei token per cluster
df_tokens = df.loc[:,tokens_clust].groupby('cluster').mean() 
#review tokens
df_review = df_tokens.iloc[:,0:63]

#print the top 20 review tokens  for cluster 4
cluster=4
print(df_review.iloc[cluster,:].sort_values(ascending=False)[0:20])

#print the top 20 review tokens for cluster 5
cluster=5
print(df_review.iloc[cluster,:].sort_values(ascending=False)[0:20])

#print the top 20 review tokens a for cluster 9
cluster=9
print(df_review.iloc[cluster,:].sort_values(ascending=False)[0:20])

tokens = ['r_excel','r_love','r_nice']
df_tokens = df[tokens+['cluster']].groupby('cluster').mean()

plt.subplots(1,3)
for token in range(len(tokens)):
    plt.subplot(1,3,token+1)
    plt.bar(df_tokens.index, df_tokens[tokens[token]])
    plt.xticks(range(10))
    plt.xlabel('clusters')
    plt.ylabel('Prominence value')
    plt.title(f'Term "{tokens[token]}" in each cluster')
plt.tight_layout()
plt.show()

tokens = ['r_employe','r_place','r_clean','r_custom r_servic','r_eat', 'r_work']
df_tokens = df[tokens+['cluster']].groupby('cluster').mean()

plt.subplots(2,3)
for token in range(len(tokens)):
    plt.subplot(2,3,token+1)
    plt.bar(df_tokens.index, df_tokens[tokens[token]])
    plt.xticks(range(10))
    plt.xlabel('clusters')
    plt.ylabel('Prominence value')
    plt.title(f'Term "{tokens[token]}" in each cluster')
plt.tight_layout()
plt.show()

