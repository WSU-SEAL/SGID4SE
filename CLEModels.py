# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Sayma Sultana <sayma@wayne.edu>, Jaydeb Sarker <jaydebsarker@wayne.edu> ,and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from nltk import word_tokenize
from sklearn_pandas import DataFrameMapper

import numpy as np
from  pprint import pprint

import  nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

mystop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ourselves', 'you', 'your',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'themselves',
                'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                'and', 'if', 'or', 'as', 'until', 'of', 'at', 'by', 'between', 'into',
                'through', 'during', 'to', 'from', 'in', 'out', 'on', 'off', 'then', 'once', 'here',
                'there', 'all', 'any', 'both', 'each', 'few', 'more',
                'other', 'some', 'such', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now']


class CLEModel:
    def __init__(self, X_train, Y_train, algo="RF", minority_bias=1):
        self.algo = algo
        self.vectorizer = TfidfVectorizer(tokenizer=word_tokenize, sublinear_tf=True, max_df=0.5,
                                          stop_words=mystop_words, min_df=20)

        self.bias = minority_bias
        self.mapper = None
        self.Y = None
        self.X = None
        self.clf = self.get_classifier()
        self.__prepare_data(X_train, Y_train)
        self.model = self.train()


    def __prepare_data(self, X_train, Y_train):
        self.mapper = DataFrameMapper([
            ('message', self.vectorizer),
            ('role_reference', None),
            ('women_kin_reference', None),
            ('lgbtq_reference', None),
            ('pejorative_count', None),
            ('appearance_reference', None),
            ('women_body_part_reference', None),
            ('women_cloth_reference', None),
        ])
        self.Y = np.ravel(Y_train)
        self.X = self.mapper.fit_transform(X_train)

    def get_classifier(self):
        algo = self.algo

        class_weight = {0: 1, 1: self.bias}

        if algo == "RF":
            return RandomForestClassifier(n_jobs=-1, min_samples_split=5, class_weight=class_weight)
        elif algo == "DT":
            return DecisionTreeClassifier()
        elif algo == "SVM":
            return LinearSVC()
        elif algo == "LR":
            return LogisticRegression()
        elif algo == "XGB":
            return GradientBoostingClassifier()


    def train(self):
        print("Training the model with " + str(len(self.Y)) + " instances and " + str(
            self.X.shape[1]) + " features")
        self.clf.fit(self.X, self.Y)
        print("Model training complete ..")
        return self.clf

    def predict(self, X_test):
        X_test_mapped = self.mapper.transform(X_test)
        predictions = self.model.predict(X_test_mapped)
        return np.expand_dims(predictions, 1)
