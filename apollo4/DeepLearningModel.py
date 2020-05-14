'''
This file contains code for all deep learning models
'''
CUDA_LAUNCH_BLOCKING="1"
import os
# install torch version 1.4.0
import torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# Simple transformers doesn't have versions. Just install simpletransformers.
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from dateutil.relativedelta import relativedelta
import datetime
from collections import Counter
import pandas as pd
from datetime import datetime
import os
import re
import codecs
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import numpy as np

def dedup_collection_journal_dl(file_open, uid, abstract_id):
    try:
        new_file_list = []
        new_item_list = []
        for doc in file_open:
            item = '\t'.join(doc.split('\t')[uid: abstract_id + 1])
            if item not in new_item_list:
                new_item_list.append(item)
                new_file_list.append(doc)
        return new_file_list
    except Exception as e:
        return "Error running the program. Please contact the IP Group Analytics Team (apolloip@ssi.samsung.com) to resolve the issue. Please provide the error details below in your email. \nPlease provide all the steps to reproduce this issue. \n" + "-" * 40 + "\n" + str(
                e) + "\n" + "-" * 40


def preprocess_data_dl(data, type_train_or_test = 'train'):
    '''
    We'll need to transform our data into a format that deep learning models understand. This involves two steps:
    def load_dataset(dataFile):
    trainingDataTypePatentOrJournal = None
    #import tensorflow as tf
    '''
    trainingDataTypePatentOrJournal = None
    file_sample_open = data.split('\n')
    fileHeader = file_sample_open[0]

    if 'identification number\ttitle\tabstract\tclaims\tapplication number\tapplication date\tcurrent assignee\tupc' in fileHeader.lower():
        # it is patent data
        trainingDataTypePatentOrJournal = 'Patent'

    elif 'meta data\ttitle\tabstract\tauthor\taffiliation\tpublished year' in fileHeader.lower():
        # it is journal data
        trainingDataTypePatentOrJournal = 'Journal'

    # The code for loading and pre-processing the data is different for patent and journal data
    if trainingDataTypePatentOrJournal == 'Patent':
        # file_sample_open = data
        # file_sample_open = file_sample_open.split('\n')  # split by new line
        file_sample_open = list(filter(None, file_sample_open))  # delete empty lines

        # The first line is header, so remove the first line from the list of documents
        file_sample_open = file_sample_open[1:]

        # Preprocess the sample file
        # file_sample_stem = preprocess_collection(dataFilePath, file_sample_open, stopterms, True)
        file_sample = list(filter(None, file_sample_open))

        title_samples = [doc.split('\t')[1] for doc in file_sample]
        abstract_samples = [doc.split('\t')[2] for doc in file_sample]
        claim_samples = [doc.split('\t')[3] for doc in file_sample]

        # Combine the text from the title, abstract, and claims columns.
        train_data = ['. '.join(doc) for doc in zip(title_samples, abstract_samples, claim_samples)]

        # This is to load the labels for Tier 1 categories. Some datasets have Tier 2 categories. Please check this code, if you use Tier 2 categories.
        if type_train_or_test == 'train':
            label_samples = [doc.split('\t')[8].lower() for doc in file_sample]
            labels = sorted(list(set(label_samples)))
            le = preprocessing.LabelEncoder()
            le.fit(label_samples)
            train_target = le.transform(label_samples)
        else:
            label_samples = [doc.split('\t')[8].lower() for doc in file_sample]
            labels = sorted(list(set(label_samples)))
            le = preprocessing.LabelEncoder()
            le.fit(label_samples)
            train_target = le.transform(label_samples)

    elif trainingDataTypePatentOrJournal == 'Journal':
        # file_sample_open = data
        # file_sample_open = file_sample_open.split('\n')  # split by new line
        file_sample_open = list(filter(None, file_sample_open))  # delete empty lines

        # Now, the first line is header, so remove the first line
        file_sample_open = file_sample_open[1:]

        # Remove the duplicated documents based on both "title" and "abstract"
        file_sample_open_training = dedup_collection_journal_dl(file_sample_open, 1, 2)

        file_sample_data = ['. '.join([doc.split('\t')[1]
                                          , doc.split('\t')[2]
                                       ]) for doc in file_sample_open_training]

        # Training Phase
        label_samples = [doc.split('\t')[-1].lower() for doc in file_sample_open]
        labels = sorted(list(set(label_samples)))

        train_data = file_sample_data
        le = preprocessing.LabelEncoder()
        le.fit(label_samples)
        train_target = le.transform(label_samples)

        #print(len(train_target))
        print("Training data size: %d documents" % len(train_data))
    return trainingDataTypePatentOrJournal, train_data, train_target, le

class DeepLearningModel():
    def __init__(self, model_type, model_name, batch_size, max_sequence_length, num_epochs, random_state, output_dir):
        self.model = None
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.random_state = random_state
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.model_type = model_type
        self.label_encoder = None
        self.output_dir = output_dir

    # The fit function first preprocesses the data and then builds the model on that data.
    def fit(self, data):
        try:
            print(111)
            trainingDataTypePatentOrJournal, train_data, train_target, le = preprocess_data_dl(data, type_train_or_test="train")

            self.label_encoder = le
            train = pd.DataFrame(list(zip(train_data, train_target)), columns = ['data', 'target'])
            train_df = pd.DataFrame(list(zip(train_data, train_target)), columns = ['text', 'label'])
            total_labels = len(Counter(train_df['label']))

            # Empty the cuda cache, if possible, to make sure there is enough GPU memory available.
            torch.cuda.empty_cache()
            print(self.model_type)
            print(self.model_name)
            print(self.max_sequence_length)
            print(self.num_epochs)
            print(self.output_dir)

            self.model = ClassificationModel(self.model_type, self.model_name, num_labels = total_labels,
                                        args={'max_seq_length': self.max_sequence_length,
                                              'training_batch_size': self.batch_size,
                                              'n_gpu': 4,
                                              "num_train_epochs": self.num_epochs,
                                              "fp16": False,
                                              "use_early_stopping": True,
                                              "sliding_window": True,
                                              "reprocess_input_data": True,
                                              "early_stopping_patience": 5,
                                              "early_stopping_delta": 0.001,
                                              "output_dir": self.output_dir},use_cuda=False)
            print(self.model)

            print(1234)
            self.model.train_model(train)
            print(9999)

        except Exception as ex:
            if 'cuda out of memory' in str(ex).lower():
                print('cuda')
                return "Out of Memory Exception"
            else:
                print('cuda else')
                return "Unknown error while training the deep learning model."
        print(12345)
        return "success"

    def classes__(self):
        if self.label_encoder != None:
            return self.label_encoder.classes__
        else:
            return None

    def predict(self, test_data):
        testingDataTypePatentOrJournal, test_data, test_target, le = preprocess_data_dl(test_data, type_train_or_test="test")
        if self.model == None:
            return "Model needs to be initialized and trained before making prediction."
        result, model_outputs, wrong_predictions = self.model.eval_model(test_data)
        test_list = test_data['text'].tolist()
        test_list_y = test_data['label'].tolist()
        predictions, raw_outputs = self.model.predict(test_list)

        # Code to convert numerical labels back to string
        predicted_labels = list(self.label_encoder.inverse_transform(predictions))
        return predicted_labels, test_list_y

    def predict_proba(self, test_data):
        testingDataTypePatentOrJournal, test_data, test_target, le = preprocess_data_dl(test_data,
                                                                                        type_train_or_test="test")
        if self.model == None:
            return "Model needs to be initialized and trained before making prediction."
        # result, model_outputs, wrong_predictions = self.model.eval_model(test_data)
        test_list = test_data['text'].tolist()
        test_list_y = test_data['label'].tolist()
        predictions, raw_outputs = self.model.predict(test_list)

        # Code to convert scores to probability values
        raw_outputs_formatted = [out.tolist()[0] for out in raw_outputs]
        raw_outputs_formatted = np.array(raw_outputs_formatted)
        predicted_probabilities = softmax(raw_outputs_formatted, axis=1)
        prediction = {
            "predicted_probabilities":predicted_probabilities,
            "test_list_y": test_list_y
        }
        return prediction
