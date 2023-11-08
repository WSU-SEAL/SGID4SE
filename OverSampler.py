# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Sayma Sultana <sayma@wayne.edu>, Jaydeb Sarker <jaydebsarker@wayne.edu> ,and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import  json
import math

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from nltk import word_tokenize

def duplicate_df_to_target(df, target_size):
    df_sampled = pd.DataFrame()
    while len(df_sampled) < target_size:
        df_sampled = df_sampled.append(df.sample(replace=True))
    return df_sampled.iloc[:target_size, :]

def read_dataframe_from_excel(file):
    dataframe = pd.read_excel(file)
    return dataframe

class SimilarWordReplacer:
    def __init__(self,  model_file="models/word-groups.json"):
        self._model_file =model_file
        self._group_mapping = self.read_word_groups()
        self._word_to_group =self.prepare_mapping(self._group_mapping)


    def read_word_groups(self):
        with open(self._model_file) as jsonfile:
            json_list =json.load(jsonfile)
            return json_list

    def prepare_mapping(self, group_list):
        word_dictionary =dict()
        for group in group_list:
                group_entries = group_list[group]
                for  entry in group_entries:
                    word_dictionary [entry]=group
        return word_dictionary

    def word_exists(self, word):
        if word in self._word_to_group:
            return True
        return False

    def get_equivalent_words(self, word):
        if not self.word_exists(word):
            return None
        group_number =self._word_to_group[word]

        word_list =self._group_mapping[group_number]
        replacements =list()

        for entry in word_list:
            if entry!= word:
                replacements.append(entry)
        return replacements

    def create_additional_samples(self, text):
        new_samples =list()

        instance_words = word_tokenize(text)
        for word in self._word_to_group.keys():
            if word in instance_words:
                alternatives =self.get_equivalent_words(word)
                for alternate in alternatives:
                    new_sentence =text.replace(word, alternate)
                    new_samples.append(new_sentence)
        return new_samples



        return count

class WordReplaceBasedOversampler:
    def  __init__(self, sampling_strategy=0.5):
        self._sampler= SimilarWordReplacer()
        self._ratio =sampling_strategy


    def get_all_synthetic_samples(self, X_df, Y_df):

        merged_df =pd.concat([X_df, Y_df], axis=1)
        new_samples =merged_df.iloc[:0,:].copy()
        positive_df =merged_df[merged_df['target']==1]
        positive_df.reset_index()
        number_positive= len(positive_df)
        total_sample =len(X_df)
        number_negative =total_sample-number_positive

        if self._ratio>1:
            target_positive =self._ratio

        else:
            target_positive = math.ceil ((number_negative* self._ratio) -number_positive)

        if target_positive<=0:
            print("Invalid ratio")
            return X_df, Y_df

        num_additional_samples=0

        for index, row in positive_df.iterrows():

            text =row['message']
            alternate_texts =self._sampler.create_additional_samples(text)

            if alternate_texts is not None:
                for new_text in alternate_texts:
                    new_row = row.copy(deep=True)
                    new_row['message']=new_text
                    new_samples =new_samples.append(new_row, ignore_index=False)
                    num_additional_samples= num_additional_samples+1

        print("created new samples "+ str(num_additional_samples))
        random_sample =new_samples.sample(n=target_positive)
        merged_df =pd.concat([merged_df, random_sample], axis=0)
        merged_df = merged_df.sample(frac=1).reset_index(drop=True)

        return merged_df

    def fit_resample(self, X_df, Y_df):

        synthetic_df =self.get_all_synthetic_samples(X_df, Y_df)

        return_x_df =synthetic_df.loc[:, synthetic_df.columns != 'target']
        return_y_df =synthetic_df["target"]


        return  return_x_df, return_y_df


class MixedOversampler:
    def  __init__(self, sampling_strategy=0.5):
        self._sampler= SimilarWordReplacer()
        self._ratio =sampling_strategy

    def fit_resample(self, X_df, Y_df):
        merged_df =pd.concat([X_df, Y_df], axis=1)

        positive_df =merged_df[merged_df['target']==1]
        positive_df.reset_index()
        number_positive= len(positive_df)
        total_sample =len(X_df)
        number_negative =total_sample-number_positive


        target_positive =int(number_negative* self._ratio)
        new_samples =math.floor((target_positive- number_positive)*0.5)

        print("Target postive samples: " + str(target_positive))

        if new_samples>0:
            oversample_df =duplicate_df_to_target(positive_df, number_positive*10)
            oversample_df= oversample_df.sample(n=new_samples).reset_index(drop=True)

            print("Duplicated total: " +str(len(oversample_df)))

            generate_sampler =WordReplaceBasedOversampler(sampling_strategy=new_samples)
            synthetic_df =generate_sampler.get_all_synthetic_samples(X_df, Y_df)

            number_synthetic =int(target_positive-(new_samples +number_positive))
            synthetic_df_samples =synthetic_df.sample(n=number_synthetic)

            print("Selected synthetic total: " +str(len(synthetic_df_samples)))

            return_df =pd.concat([merged_df, oversample_df, synthetic_df_samples], axis=0)
            return_df = return_df.sample(frac=1).reset_index(drop=True) #randomize
            return_x_df =return_df.loc[:, return_df.columns != 'target']
            return_y_df =return_df["target"]

            return  return_x_df, return_y_df











#
#
# gsdset= GSD4SE()
#
# dataframe =gsdset.get_training_data()
#
# kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
# results = ""
#
#
# custom_sampler =WordReplaceBasedOversampler(sampling_strategy=0.7)
#
# i=1
#
# for train_index, test_index in kf.split(dataframe, dataframe["target"]):
#     print("Iteration"+ str(i))
#
#     X_train, X_test = dataframe.loc[train_index, ["message", "role_reference","maternal_reference","gender_reference",
#                             "adjective_count","physical_reference", "CommentID"]], \
#                       dataframe.loc[test_index, ["message", "role_reference","maternal_reference","gender_reference",
#                             "adjective_count","physical_reference", "CommentID"]]
#     Y_train, Y_test = dataframe.loc[train_index, "target"], dataframe.loc[test_index, "target"]
#
#     X_augment, Y_augment =custom_sampler.fit_resample(X_train, Y_train)
#
#     print(len(X_augment))
#     print(len(Y_augment))
#     i=i+1












