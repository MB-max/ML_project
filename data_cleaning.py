import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

#importing data
df = pd.read_csv("McDonald_s_Reviews.csv", encoding="latin-1")
print(df.head())
#check the data types
print(df.info() )


print(df.isnull().sum() )
print(df.duplicated().sum())

print(df.columns)
for column in df.columns:
    num_distinct_values = len(df[column].unique())
    print(f"{column}: {num_distinct_values} distinct values")
    
    
print(df[df.isnull().any(axis = 1)]) 
print(df.describe() )
print(df["rating"].value_counts() )

df1 = df.copy()
df1 = df1.drop(columns=["category", "store_address", "latitude ", "longitude", "review_time"])
print(df1.describe() )

plt.figure(figsize=(15,8))
df1["rating"] = df1["rating"].str.split(" ").str[0]
print(df1["rating"])
df1 = df1.astype({'rating':'int'})
print(df1.info() )
labels = df1["rating"].value_counts().keys()
values = df1["rating"].value_counts().values
explode = (0.1,0,0,0,0)
plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.title('Proportion of each rating',fontweight='bold',fontsize=25,pad=20,color='crimson')
plt.show() 

def clean_review(review):
    review = review.lower()
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = re.sub(r'\s+', ' ', review).strip()

    stop_words = set(stopwords.words('english'))
    review_tokens = nltk.word_tokenize(review)
    review = ' '.join([word for word in review_tokens if word not in stop_words])

   
    return review

df1['clean_reviews'] = df1['review'].apply(clean_review)
df1 = df1.drop(columns=['review'])
print(df1[['clean_reviews']])

stemmer = PorterStemmer()
def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])
df1['clean_reviews'] = df1['clean_reviews'].apply(lambda x: stem_words(x))

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
df1["clean_reviews"] = df1["clean_reviews"].apply(lambda text: lemmatize_words(text))

df1['clean_reviews'].head()
df1.to_csv('Preprocessed_McDonald_s_Reviews.csv')
print(df1[df1.isnull().any(axis = 1)])
print(df1.isnull().sum() )