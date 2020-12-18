# -*- coding: utf-8 -*-

import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from lime import lime_text
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

import re

from keras.models import Sequential, Model, load_model
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, Activation, Input, Reshape, Concatenate
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical

def __payload_segmentation_str(raw_string, split_exp):
    """String segmentaion function

    Get string segmented with given regular expressions for text preprocessing.

    Args:
        raw_string: strings to be preprocessed.
        split_exp: configured regular expression.

    Returns:
        Segmented texts.
    """

    splitter = re.compile(r'(%s)|$' % split_exp)
    text = [s for s in splitter.split(raw_string) if len(s)>=2] # Remove single charaters
    text = ' '.join(text)
    return text

def text_preprocess(df, x_col='text', y_col='target'):
    """Preprocessing for texts.

    Some preprocssing procedure for text in a dataframe, such as unreadable character
    revomval, label encoding and so on.

    Args:
        df: datafame to be preprocessed.
        x_col: the text column for processing.
        y_col: the label column for processing.

    Returns:
        df: preprocessed dataframe
        label_map: the target label and encoding result mapping
    """

    # remove unreadable character
    df[x_col] = df[x_col].str.replace(r'[^\x00-\x7F]+', ' ')
    df = df.dropna()

    # reset index
    df.index = np.arange(1, len(df) + 1)

    # label encoding
    le = preprocessing.LabelEncoder()
    le.fit(df[y_col])
    df[y_col] = le.transform(df[y_col]) # example for binary classification {'malicious': 0, 'normal': 1}

    # get label and value mapping
    label_map = dict(zip(le.classes_, range(len(le.classes_))))
    
    return df, label_map
        
class text_model_generator(object):
    """Generate model for text classification for test only. Model is not optimized and well-chosen.

    Attributes:
        df: train dataset dataframe
        x_col, y_col: 'text' and 'target' respectively.
        X, Y: Serias for 'text' and 'target' respectively.
        X_train, X_val, Y_train, Y_val: Generated train and validation datasets.
    """

    def __init__(self, df, x_col='text', y_col='target'):
        """Inits with inputs."""
        self.df = df
        self.x_col = x_col
        self.y_col = y_col

        self.X = df[self.x_col]
        self.Y = df[self.y_col]
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X, self.Y, test_size = 0.2, random_state = 1)
    
    def get_labelencoder(self):
        """Get label encoder."""
        labelencoder = preprocessing.LabelEncoder()
        return labelencoder

    def get_vectorizer(self):
        """Get Tfidf Vectorizer."""
        vectorizer = TfidfVectorizer(lowercase=False)
        print(vectorizer)
        return vectorizer
    
    def get_model(self):
        """Get classification model."""
        model = MultinomialNB(alpha=.01)
        return model
    
    def get_pipeliner(self):
        """Make modle pipeline."""
        vectorizer  = self.get_vectorizer()
        model = self.get_model()

        pipeliner = make_pipeline(vectorizer, model)
        return pipeliner
    
    def model_trainer(self):
        """Get the pipeline model trained."""
        pipeliner = self.get_pipeliner()
        pipeliner.fit(self.X_train, self.Y_train)

        pred = pipeliner.predict(self.X_val)
        print(pred)
        print(type(pred))
        print(pred.shape)
        print(sklearn.metrics.f1_score(self.Y_val, pred, average='binary', pos_label=0))

        return pipeliner