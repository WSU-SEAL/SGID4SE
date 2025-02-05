# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Sayma Sultana <sayma@wayne.edu>, Jaydeb Sarker <jaydebsarker@wayne.edu> ,and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import os.path
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import KFold, StratifiedKFold

from ContractionPreprocessor import expand_contraction, rem_special_sym, remove_url
from OverSampler import WordReplaceBasedOversampler, MixedOversampler
from PatternProcessor import PatternTokenizer, IdentifierTokenizer
from CLEModels import CLEModel
from sklearn.metrics import classification_report
import argparse
import warnings
import random
import timeit

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def read_dataframe_from_excel(file):
    dataframe = pd.read_excel(file)
    return dataframe


class SGID4SE:
    def __init__(self, ALGO="BERT", model_file="models/SGID-dataset-full.xlsx", split_identifier=True,
                 remove_keywords=True,  load_pretrained=False, count_words =False):
        self.classifier_model = None
        self.modelFile = model_file
        self.split_identifier = split_identifier
        self.remove_keywords = remove_keywords
        self.source_code_checker = IdentifierTokenizer()
        self.pattern_checker = PatternTokenizer()
        self.ALGO = ALGO
        self.training_data = read_dataframe_from_excel(model_file)
        self.load_pretrained = load_pretrained
        self.count_words = count_words

    def preprocess(self, dataframe):
        dataframe["message"] = dataframe.message.astype(str).apply(self.process_text)

        if self.count_words:
            dataframe["role_reference"] = dataframe.message.astype(str). \
                apply(self.pattern_checker.count_women_roles)
            dataframe["women_kin_reference"] = dataframe.message.astype(str). \
                apply(self.pattern_checker.count_women_kins_reference)
            dataframe["lgbtq_reference"] = dataframe.message.astype(str). \
                apply(self.pattern_checker.count_lgbtq_reference)
            dataframe["pejorative_count"] = dataframe.message.astype(str). \
                apply(self.pattern_checker.count_pejoratives)
            dataframe["appearance_reference"] = dataframe.message.astype(str). \
                apply(self.pattern_checker.count_appearance_reference)
            dataframe["women_body_part_reference"] = dataframe.message.astype(str). \
                apply(self.pattern_checker.count_women_body_parts)
            dataframe["women_cloth_reference"] = dataframe.message.astype(str). \
                apply(self.pattern_checker.count_women_clothes)
        else:
            dataframe["role_reference"] =0
            dataframe["women_kin_reference"] = 0
            dataframe["lgbtq_reference"] = 0
            dataframe["pejorative_count"] = 0
            dataframe["appearance_reference"] = 0
            dataframe["women_body_part_reference"] = 0
            dataframe["women_cloth_reference"] = 0


    def get_training_data(self):
        self.preprocess(self.training_data)
        return self.training_data

    def __get_pretrained_model(self):
        return True

    def process_text(self, text):
        processed_text=text
        # mandatory preprocessing
        processed_text = remove_url(processed_text)
        processed_text = expand_contraction(processed_text)
        processed_text = self.pattern_checker.process_text(processed_text)
        processed_text = rem_special_sym(processed_text)
        # optional preprocessing
        if self.split_identifier:
            processed_text = self.source_code_checker.split_identifiers(processed_text)

        if self.remove_keywords:
            processed_text = self.source_code_checker.remove_keywords(processed_text)
        return processed_text

    def init_predictor(self, strategy ="random", ratio =1):
        if self.load_pretrained:
            filename = self.getPTMName()
            loadstatus = self.load_pretrained_model(filename)
            if loadstatus:
                print("Successfully loaded pretrained model from "+filename)
                return
            else:
                print("Unable to load pretrained model.."+filename)
        self.__train_predictor( strategy, ratio)

    def getPTMName(self):
        ALGO=self.ALGO
        filename = "./pre-trained/model-" + ALGO \
                   + "-keyword-" + str(self.remove_keywords) + "-split-" \
                   + str(self.split_identifier)
        if ((ALGO == "CNN") | (ALGO == "LSTM") | (ALGO == "GRU")  ):
            filename = filename + ".h5"
        elif((ALGO =="BERT") | (ALGO =="ALBERT")| (ALGO =="SBERT")):
            filename = filename + ".h5"
        elif ((ALGO == "RF") | (ALGO == "SVM") | (ALGO == "DT") | (ALGO == "LR") | (ALGO == "XGB") ):
            filename = filename + ".pickle"
        return filename

    def __train_predictor(self, strategy ="random", ratio =1 ):
        self.preprocess(self.training_data)
        X_train = self.training_data[["message", "role_reference","women_kin_reference","lgbtq_reference",
                                "pejorative_count","appearance_reference", "women_body_part_reference",
                                      "women_cloth_reference",  "CommentID"]]
        Y_train = self.training_data[['target']]
        # train model using full dataset



        if strategy=="random":
            oversampler = RandomOverSampler(sampling_strategy=ratio)
        elif strategy=="generate":
            oversampler =WordReplaceBasedOversampler(sampling_strategy=ratio)
        elif strategy=="mixed":
            oversampler =MixedOversampler(sampling_strategy=ratio)

        X_train, Y_train = oversampler.fit_resample(X_train, Y_train)

        self.get_model(X_train, Y_train)


    def save_trained_model(self):
        ALGO = self.ALGO
        filename = self.getPTMName()
        if ((ALGO == "BERT") | (ALGO == "CNN") | (ALGO == "LSTM") | \
                (ALGO == "GRU") | (ALGO == "ALBERT")| (ALGO == "SBERT")):
            self.classifier_model.save_to_file(filename)
        elif ((ALGO == "RF") | (ALGO == "SVM") | (ALGO == "DT") | (ALGO == "LR")):
            pickle.dump(self.classifier_model, open(filename, "wb"))
        print("Model stored as: "+filename)

    def load_pretrained_model(self, filename):
        if not os.path.exists(filename):
            print("File: "+ filename +" not exists!")
            return False
        if filename.endswith(".pickle"):
            self.classifier_model = pickle.load(open(filename, "rb"))
            return True
        ALGO = self.ALGO
        try:
            if ((ALGO == "CNN") | (ALGO == "GRU") | (ALGO == "LSTM")):
                import DNNModels
                self.classifier_model = DNNModels.DNNModel(algo=ALGO, load_from_file=filename)
                return True
            elif ((ALGO == "BERT" )| (ALGO == "ALBERT") | (ALGO == "SBERT")):
                from TransformerModel import TransformerModel
                self.classifier_model = TransformerModel(load_from_file=filename, model_name=ALGO)
                return True
        except Exception as e:
            print(e)
            return False

    def get_model(self, X_train, Y_train, minority_bias=1):
        ALGO = self.ALGO
        if (ALGO == "RF")  | (ALGO == "SVM") | (ALGO == "DT") | (ALGO == "LR") | (ALGO =="XGB"):
            self.classifier_model = CLEModel(X_train=X_train, Y_train=Y_train, algo=self.ALGO, minority_bias= minority_bias,)
        elif  (ALGO == "BERT") | (ALGO == "ALBERT") | (ALGO == "SBERT") :
            from TransformerModel import TransformerModel
            self.classifier_model = TransformerModel(X_train=X_train, Y_train=Y_train, model_name=ALGO,
                                                     minority_bias=minority_bias)
        elif (ALGO == "CNN") | (ALGO == "LSTM") | (ALGO == "GRU") :
            import DNNModels

            self.classifier_model = DNNModels.DNNModel(X_train=X_train,
                                                       Y_train=Y_train, minority_bias= minority_bias, algo=ALGO )
        else:
            print("Unknown algorithm: "+ALGO)
            exit(1)

        return self.classifier_model

    def get_SGID_probablity(self, texts):
        dataframe = pd.DataFrame(texts, columns=['message'])
        self.preprocess(dataframe)
        #print(dataframe)
        results = self.classifier_model.predict(dataframe)
        return results



def get_detail_classifications(dataframe, labels, Y_prob, predictions):
    predictions = pd.DataFrame(data=predictions, columns=["predicted"])
    probabilities = pd.DataFrame(data=Y_prob, columns=["probability"])

    newdf = dataframe.reset_index(drop=True)
    labels_reset = labels.reset_index(drop=True)
    merged_df = pd.concat([newdf, predictions], axis=1)
    merged_df = pd.concat([merged_df, labels_reset], axis=1)
    merged_df = pd.concat([merged_df, probabilities], axis=1)

    return merged_df


def binary_class_by_threshold(predicttion_probablities, threshold=0.5):
    predictions =[1 if pred >= threshold else 0 for pred in predicttion_probablities]
    return predictions

def ten_fold_cross_validation(amiClassifier, oversample ="random", ratio=0, bias=1, threshold_vary =False,  rand_state =55):
    dataset = amiClassifier.get_training_data()
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rand_state)
    results = ""


    count = 1
    all_predictions = pd.DataFrame()

    for train_index, test_index in kf.split(dataset, dataset["target"]):
        start = timeit.default_timer()
        print("Using split-" + str(count) + " as test data..")


        X_train, X_test = dataset.loc[train_index, ["message", "role_reference","women_kin_reference","lgbtq_reference",
                                "pejorative_count","appearance_reference", "women_body_part_reference",
                                      "women_cloth_reference", "CommentID"]], \
                          dataset.loc[test_index, ["message", "role_reference","women_kin_reference","lgbtq_reference",
                                "pejorative_count","appearance_reference", "women_body_part_reference",
                                      "women_cloth_reference", "CommentID"]]
        Y_train, Y_test = dataset.loc[train_index, "target"], dataset.loc[test_index, "target"]

        #if append:
        #    X_extra =extra_train_df[["message"]]
        #    Y_extra =extra_train_df[["target"]].squeeze()

        #    X_train =pd.concat([X_train,X_extra], axis=0, ignore_index=True)
        #    Y_train =pd.concat([Y_train, Y_extra], axis=0, ignore_index=True)



        if ratio!=0:
            strategy =1.0/ratio
            #oversampler = RandomOverSampler(sampling_strategy=ratio)
            if oversample=="random":
                oversampler = RandomOverSampler(sampling_strategy=strategy)
            elif oversample=="generate":
                oversampler =WordReplaceBasedOversampler(sampling_strategy=strategy)
            elif oversample=="mixed":
                oversampler =MixedOversampler(sampling_strategy=strategy)
            X_train, Y_train = oversampler.fit_resample(X_train, Y_train) #resampling only the training data

        classifier_model = amiClassifier.get_model(X_train, Y_train, minority_bias=bias)

        Y_prob = classifier_model.predict(X_test)

        if not threshold_vary:
            results = results + str(count) + "," + ALGO + ","
            predictions =binary_class_by_threshold(Y_prob)
            detail_Classification = get_detail_classifications(X_test, Y_test, Y_prob, predictions)

            stop = timeit.default_timer()
            time_elapsed = stop - start

            if len(all_predictions.columns) == 0:
                all_predictions = detail_Classification

            else:
                all_predictions = pd.concat([all_predictions, detail_Classification], axis=0)


            precision_1 = precision_score(Y_test, predictions, pos_label=1)
            recall_1 = recall_score(Y_test, predictions, pos_label=1)
            f1score_1 = f1_score(Y_test, predictions, pos_label=1)

            precision_0 = precision_score(Y_test, predictions, pos_label=0)
            recall_0 = recall_score(Y_test, predictions, pos_label=0)
            f1score_0 = f1_score(Y_test, predictions, pos_label=0)
            accuracy = accuracy_score(Y_test, predictions)
            mcc = matthews_corrcoef(Y_test, predictions)

            results = results + str(precision_0) + "," + str(recall_0) + "," + str(f1score_0)
            results = results + "," + str(precision_1) + "," + str(recall_1) + "," + str(f1score_1) + \
                      "," + str(accuracy) + "," + str(mcc) +  "," + str(time_elapsed) + "\n"

            print(classification_report(Y_test, predictions))
            count += 1
        else:
            best_threshold=0.05
            best_fscore=0.00
            best_mcc=0.00
            best_mcc_threshold=0.00

            for probablity in np.arange(0.01,1.00, 0.01):
                predictions =binary_class_by_threshold(Y_prob, probablity)

                precision_1 = precision_score(Y_test, predictions, pos_label=1)
                recall_1 = recall_score(Y_test, predictions, pos_label=1)
                f1score_1 = f1_score(Y_test, predictions, pos_label=1)

                precision_0 = precision_score(Y_test, predictions, pos_label=0)
                recall_0 = recall_score(Y_test, predictions, pos_label=0)
                f1score_0 = f1_score(Y_test, predictions, pos_label=0)
                accuracy = accuracy_score(Y_test, predictions)
                mcc = matthews_corrcoef(Y_test, predictions)

                results = results + str(count) +"," + str(precision_0) + "," + str(recall_0) + "," + str(f1score_0)
                results = results + "," + str(precision_1) + "," + str(recall_1) + "," + str(f1score_1) + \
                          "," + str(accuracy) + "," + str(mcc) +  "," + str(probablity) + "\n"

                if f1score_1>best_fscore:
                        best_threshold=probablity
                        best_fscore=f1score_1
                if best_mcc<mcc:
                        best_mcc_threshold=probablity
                        best_mcc=mcc
            print("Best f1 : "+ str(best_fscore)+ "at threshold: "+str(best_threshold))
            print("Best mcc: "+ str(best_mcc)+ "at threshold: "+str(best_mcc_threshold))
            count += 1

    return (results, all_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGID4SE: Sexual orientation and Gender identity Discriminatory\n'+
                                                 ' Text Detector for the SE domain')

    parser.add_argument('--algo', type=str,
                        help='Classification algorithm. Choices are: RF| DT| SVM| LR| CNN| LSTM| GRU| BERT| ALBERT| SBERT',
                        default="RF")

    parser.add_argument('--oversample', type=str, help='Oversampling strategy:  '+
                                                   'random | generate | mixed', default="random")
    parser.add_argument('--ratio', type=float, help='Oversample minority class before training. Argument:  '+
                                                   'majority/minority ratio', default=1)
    parser.add_argument('--bias', type=int, help='Class weighting bias' +
                                                       'Minority multiply factor', default=1)
    parser.add_argument('--repeat', type=int, help='Iteration count', default=1)

    parser.add_argument('--split', help='Split identifiers', action='store_false', default=True)
    parser.add_argument('--wc', help='Count word features', action='store_true', default=False)
    parser.add_argument('--vary', help='Experment threshold variation', action='store_true', default=False)

    parser.add_argument('--keyword', help='Remove programming keywords', action='store_false', default=True)
    parser.add_argument('--retro', help='Print missclassifications',
                        action='store_true', default=False)  # default False, will not write
    parser.add_argument('--mode', type=str,
                        help='Execution mode. Choices are: eval | pretrain ',
                        default="eval")

    args = parser.parse_args()

    print(args)
    ALGO = str(args.algo).upper()
    REPEAT = args.repeat
    mode = args.mode
    oversample=args.oversample
    ratio=args.ratio
    bias = args.bias
    wordcount=args.wc
    threshold_vary =args.vary
    amiClassifier = SGID4SE(split_identifier=args.split, remove_keywords=args.keyword,
                            ALGO=ALGO, count_words=wordcount)

    if mode == 'pretrain':
        amiClassifier.init_predictor()
        amiClassifier.save_trained_model()
        exit(0)

    timers = []

    filename = "results/cross-validation-" +str(ALGO) + "-oversample-" + str(args.oversample)+"-ratio-" \
                + str(args.ratio) +"-bias-" + str(args.bias) +"-wc-" +str(args.wc) + ".csv"

    if args.vary:
        filename = "results/vary-evaluation-" +str(ALGO) + "-oversample-" + str(args.oversample)+"-ratio-" \
                + str(args.ratio) +"-bias-" + str(args.bias) +"-wc-" +str(args.wc) + ".csv"

    training_log = open(filename, 'w')
    if threshold_vary:
        training_log.write("Fold,precision_0,recall_0,f-score_0,precision_1,recall_1,f-score_1,accuracy,mcc,threshold\n")
    else:
        training_log.write("Fold,Algo,precision_0,recall_0,f-score_0,precision_1,recall_1,f-score_1,accuracy,mcc,time\n")

    random.seed(200)
    for k in range(0, REPEAT):
        print(".............................")
        print("Run# {}".format(k))
        (results, misclassified) = ten_fold_cross_validation(amiClassifier, oversample,ratio, bias, threshold_vary,  random.randint(1, 10000))
        training_log.write(results)
        training_log.flush()
        if (args.retro & (k == 0)):
            misclassified.to_excel("results/"+ str(ALGO) + "-oversample-" + str(args.oversample)+
                                  "-ratio-" + str(args.ratio)
                                   + "_misclassified.xlsx")

    ##########################

    training_log.close()
