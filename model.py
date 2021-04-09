import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, f1_score,accuracy_score
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import re
import scipy as sp
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('/Users/apple/Desktop/job/Zuellig Pharma/gender/name_gender/name_gender.csv',engine='python')

#preprocessing
def preprocessing(s):
    # 1.to lower
    s=s.lower()
    # 2: keep only the letter 
    s =''.join([w for w in s if w.isalpha()])
    return s

df['clean_name']=df['name'].apply(preprocessing)
df['name_len']=df['clean_name'].apply(lambda x: len(x))

#new features
df['last_letter']=df['clean_name'].apply(lambda x: x[-1])
df['last_2_letter']=df['clean_name'].apply(lambda x:x[-2:])
df['last_3_letter']=df['clean_name'].apply(lambda x:x[-3:])
df['ild']=df['clean_name'].apply(lambda x: bool(re.search('lid',x)))
df['lin']=df['clean_name'].apply(lambda x: bool(re.search('lin',x)))

features_col=['last_letter','last_2_letter','last_3_letter']
onehot=OneHotEncoder(handle_unknown='ignore') 
new_cols = onehot.fit_transform(df[features_col]).toarray()
names_col=onehot.get_feature_names(features_col)
df_new = pd.DataFrame(new_cols, dtype=bool, columns=names_col)
df=pd.concat([df,df_new],axis=1).drop(columns=features_col)

X = df.drop(columns=['gender','name','name_len'])
y = df['gender']

def combine_features(X,X_cv):
    X_manual = X.drop(columns=['clean_name'])
    X_manual=  X_manual.fillna(0)
    X_manual_sparse = sp.sparse.csr_matrix(X_manual)
    X_full = sp.sparse.hstack([X_cv, X_manual_sparse])
    return X_full

vect_tfv=TfidfVectorizer(analyzer='char',ngram_range=(2,6))
over = RandomOverSampler(random_state=1004)
X_tfv = vect_tfv.fit_transform(X['clean_name'])
X_full = combine_features(X, X_tfv)
X_full_over, y_over = over.fit_resample(X_full, y)
lr_select = LogisticRegression(random_state=1004)
lr_select.fit(X_full_over,y_over)

# Save the model
import joblib
joblib.dump(lr_select, 'model.pkl')

#load the model
lr = joblib.load('model.pkl')

#to transform input 
def to_df(s):
    s = ''.join([w for w in s if w.isalpha()])
    df = pd.DataFrame([s], columns=['clean_name'])
    return df
    
def process_input(s):
    df= to_df(s)
    df['last_letter']=df['clean_name'].apply(lambda x: x[-1])
    df['last_2_letter']=df['clean_name'].apply(lambda x:x[-2:])
    df['last_3_letter']=df['clean_name'].apply(lambda x:x[-3:])
    df['ild']=df['clean_name'].apply(lambda x: bool(re.search('lid',x)))
    df['lin']=df['clean_name'].apply(lambda x: bool(re.search('lin',x)))
    features_col=['last_letter','last_2_letter','last_3_letter']
    new_cols = onehot.transform(df[features_col]).toarray()
    names_col=onehot.get_feature_names(features_col)
    df_new= pd.DataFrame(new_cols,dtype=bool, columns=names_col)
    df=pd.concat([df,df_new],axis=1).drop(columns=features_col)
    X_tfv = vect_tfv.transform(df['clean_name'])
    X_full = combine_features(df, X_tfv)
    return X_full
