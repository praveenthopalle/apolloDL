from array import array

__version__ = '3.0'
__author__ = 'Manali Sharma'

'''
This file contains the code to optimize the parameters for various models.
'''
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import apollo4.globals

import multiprocessing as mp
from multiprocessing import Process, Lock, Queue, Value
from sklearn.model_selection import train_test_split
from apollo4.DeepLearningModel import DeepLearningModel
from collections import Counter


def getOptimalParameterForMNB_alpha_multiprocess(train_tfidf, train_data, train_target, target_performance,
                                                 candidate_alpha, process_number, queue):
    saveDatasetIndices = True
    # print "Process " + str(process_number) + "started"

    optimal_alpha = -1
    max_performance_measure = -1
    standard_deviation = -1
    all_measures_performances = []
    all_measures_standardDeviations = []

    fiveFoldModels = []
    fiveFoldTrainingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTestingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTrainingLabels = []  # save the 5 fold training labels
    fiveFoldTestingLabels = []  # Save the 5 fold testing labels

    # Do 5-folds cross validation on training data to determine the best C value
    skf = StratifiedKFold(n_splits=5, random_state=9876, shuffle=True)
    target_performances_of_all_folds = []

    # Store ALL the average performance measures (Accuracy, AUC, macro precision, macro recall, macro F1) to display in the GUI
    performances_all_measures_all_folds = []
    standard_deviations_all_measures_all_folds = []

    fiveFoldModelsCandidate = []

    for train_indices, test_indices in skf.split(train_tfidf, train_target):
        # Note: It is okay to use tf-idf transformed data for doing CV, because tf-idf is unsupervised
        X_train, X_test = train_tfidf[train_indices], train_tfidf[test_indices]
        y_train, y_test = np.array(train_target)[train_indices], np.array(train_target)[test_indices]

        X_train_original, X_test_original = [train_data[i] for i in train_indices], [train_data[j] for j in
                                                                                     test_indices]

        model_fold = MultinomialNB(alpha=candidate_alpha).partial_fit(X_train, y_train, classes=np.unique(y_train))

        # Save the indices for only one candidate_alpha; indices do not change with different candidate_alphas!
        if saveDatasetIndices == True:
            fiveFoldTrainingDatasetTfidf.append(X_train_original)
            fiveFoldTestingDatasetTfidf.append(X_test_original)
            fiveFoldTrainingLabels.append(y_train)
            fiveFoldTestingLabels.append(y_test)

        fiveFoldModelsCandidate.append(model_fold)

        (accu, auc, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1,
         pred_y) = evaluate_model_MS(model_fold, X_test, y_test, list(set(y_train)))

        if target_performance == 'accuracy':
            target_performances_of_all_folds.append(accu)
        elif target_performance == 'macro_f1':
            target_performances_of_all_folds.append(macro_f1)
        elif target_performance == 'macro_precision':
            target_performances_of_all_folds.append(macro_precision)
        elif target_performance == 'macro_recall':
            target_performances_of_all_folds.append(macro_recall)
        elif target_performance == 'auc':
            target_performances_of_all_folds.append(auc)

        performances_all_measures_one_fold = []
        performances_all_measures_one_fold.append(accu)
        performances_all_measures_one_fold.append(auc)
        performances_all_measures_one_fold.append(macro_precision)
        performances_all_measures_one_fold.append(macro_recall)
        performances_all_measures_one_fold.append(macro_f1)

        performances_all_measures_all_folds.append(performances_all_measures_one_fold)

        # Update progressbar to show progress of each fold; increment the progressbar value by 1 each time in this for loop
        # progress_value += 1

    saveDatasetIndices = False

    # Take an average of target_performances_of_all_folds
    if max_performance_measure <= np.average(target_performances_of_all_folds):
        max_performance_measure = np.average(target_performances_of_all_folds)
        standard_deviation = np.std(target_performances_of_all_folds)
        optimal_alpha = candidate_alpha

        # Additionally, store the average performance and standard deviations of all measures
        all_measures_performances = np.average(performances_all_measures_all_folds, axis=0)
        all_measures_standardDeviations = np.std(performances_all_measures_all_folds, axis=0)

        fiveFoldModels = fiveFoldModelsCandidate

    final_res = (optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
                 all_measures_standardDeviations)
    queue.put([optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
               all_measures_standardDeviations])
    return


def getOptimalParameterForMNB_alpha(train_tfidf, train_data, train_target, target_performance):
    """
    Function to compute optimal parameter, alpha, for MNB model
    """
    all_candidates_alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    processes = []
    queue = mp.Queue()
    results = np.ndarray(len(all_candidates_alpha), dtype=object)
    for i in range(len(all_candidates_alpha)):
        arguments = train_tfidf, train_data, train_target, target_performance, all_candidates_alpha[i], i, queue
        p = Process(target=getOptimalParameterForMNB_alpha_multiprocess, args=arguments)
        p.daemon = True
        processes.append(p)
        processes[i].start()
    for i in range(len(all_candidates_alpha)):
        processes[i].join()
    for i in range(len(all_candidates_alpha)):
        results[i] = queue.get()
        # Compare which candidate alpha gives the maximum performance
    max_performances = []
    for result in results:
        max_performances.append(result[1])

    optimal_alpha_index = np.argmax(max_performances)

    final_result = results[optimal_alpha_index]
    optimal_alpha = final_result[0]
    max_performance_measure = final_result[1]
    standard_deviation = final_result[2]
    all_measures_performances = final_result[3]
    all_measures_standardDeviations = final_result[4]

    return optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations


def getOptimalParameterForLR_alpha_multiprocess(train_tfidf, train_data, train_target, target_performance,
                                                candidate_alpha, process_number, queue):
    saveDatasetIndices = True
    # print "Process " + str(process_number) + "started"
    class_weight = 'balanced'

    optimal_alpha = -1
    max_performance_measure = -1
    standard_deviation = -1
    all_measures_performances = []
    all_measures_standardDeviations = []

    fiveFoldModels = []
    fiveFoldTrainingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTestingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTrainingLabels = []  # save the 5 fold training labels
    fiveFoldTestingLabels = []  # Save the 5 fold testing labels

    # Do 5-folds cross validation on training data to determine the best C value
    skf = StratifiedKFold(n_splits=5, random_state=9876, shuffle=True)
    target_performances_of_all_folds = []

    # Store ALL the average performance measures (Accuracy, AUC, macro precision, macro recall, macro F1) to display in the GUI
    performances_all_measures_all_folds = []
    standard_deviations_all_measures_all_folds = []

    fiveFoldModelsCandidate = []

    for train_indices, test_indices in skf.split(train_tfidf, train_target):
        # Note: It is okay to use tf-idf transformed data for doing CV, because tf-idf is unsupervised
        X_train, X_test = train_tfidf[train_indices], train_tfidf[test_indices]
        y_train, y_test = np.array(train_target)[train_indices], np.array(train_target)[test_indices]

        X_train_original, X_test_original = [train_data[i] for i in train_indices], [train_data[j] for j in
                                                                                     test_indices]

        random_state = np.random.RandomState(seed=3456)

        # When using SGD, the partial_fit method has to be applied on different batches of the training data, and we need to epoch multiple times
        model_fold = SGDClassifier(loss='log', penalty='l2', alpha=candidate_alpha, random_state=random_state)

        def batches(l, n):
            for i in np.arange(0, len(l), n):
                yield l[i:i + n]

        n_iter = 25
        np.random.seed(5647)
        shuffledRange = np.arange(len(X_train))
        for n in np.arange(n_iter):
            np.random.shuffle(shuffledRange)
            shuffled_X_train = [X_train[i] for i in shuffledRange]
            shuffled_y_train = [y_train[i] for i in shuffledRange]

            # Training the model in 10 batches
            for batch in batches(np.arange(len(shuffled_y_train)), 5):
                model_fold.partial_fit(shuffled_X_train[batch[0]:batch[-1] + 1],
                                       shuffled_y_train[batch[0]:batch[-1] + 1], classes=np.unique(y_train))

        # Save the indices for only one candidate_alpha; indices do not change with different candidate_alphas!
        if saveDatasetIndices == True:
            fiveFoldTrainingDatasetTfidf.append(X_train_original)
            fiveFoldTestingDatasetTfidf.append(X_test_original)
            fiveFoldTrainingLabels.append(y_train)
            fiveFoldTestingLabels.append(y_test)

        fiveFoldModelsCandidate.append(model_fold)

        (accu, auc, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1,
         pred_y) = evaluate_model_MS(model_fold, X_test, y_test, list(set(y_train)))

        if target_performance == 'accuracy':
            target_performances_of_all_folds.append(accu)
        elif target_performance == 'macro_f1':
            target_performances_of_all_folds.append(macro_f1)
        elif target_performance == 'macro_precision':
            target_performances_of_all_folds.append(macro_precision)
        elif target_performance == 'macro_recall':
            target_performances_of_all_folds.append(macro_recall)
        elif target_performance == 'auc':
            target_performances_of_all_folds.append(auc)

        performances_all_measures_one_fold = []
        performances_all_measures_one_fold.append(accu)
        performances_all_measures_one_fold.append(auc)
        performances_all_measures_one_fold.append(macro_precision)
        performances_all_measures_one_fold.append(macro_recall)
        performances_all_measures_one_fold.append(macro_f1)

        performances_all_measures_all_folds.append(performances_all_measures_one_fold)

        # Update progressbar to show progress of each fold; increment the progressbar value by 1 each time in this for loop
        # progress_value += 1

    saveDatasetIndices = False

    # Take an average of target_performances_of_all_folds
    if max_performance_measure <= np.average(target_performances_of_all_folds):
        max_performance_measure = np.average(target_performances_of_all_folds)
        standard_deviation = np.std(target_performances_of_all_folds)
        optimal_alpha = candidate_alpha

        # Additionally, store the average performance and standard deviations of all measures
        all_measures_performances = np.average(performances_all_measures_all_folds, axis=0)
        all_measures_standardDeviations = np.std(performances_all_measures_all_folds, axis=0)

        fiveFoldModels = fiveFoldModelsCandidate

    final_res = (optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
                 all_measures_standardDeviations)
    queue.put([optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
               all_measures_standardDeviations])
    return


def getOptimalParameterForLR_alpha(train_tfidf, train_data, train_target, target_performance):
    """
    Function to compute optimal parameter, alpha, for MNB model
    """
    all_candidates_alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    processes = []
    queue = mp.Queue()
    results = np.ndarray(len(all_candidates_alpha), dtype=object)
    for i in range(len(all_candidates_alpha)):
        arguments = train_tfidf, train_data, train_target, target_performance, all_candidates_alpha[i], i, queue
        p = Process(target=getOptimalParameterForLR_alpha_multiprocess, args=arguments)
        p.daemon = True
        processes.append(p)
        processes[i].start()
    for i in range(len(all_candidates_alpha)):
        processes[i].join()
    for i in range(len(all_candidates_alpha)):
        results[i] = queue.get()
        # Compare which candidate alpha gives the maximum performance
    max_performances = []
    for result in results:
        max_performances.append(result[1])

    optimal_alpha_index = np.argmax(max_performances)

    final_result = results[optimal_alpha_index]
    optimal_alpha = final_result[0]
    max_performance_measure = final_result[1]
    standard_deviation = final_result[2]
    all_measures_performances = final_result[3]
    all_measures_standardDeviations = final_result[4]

    return optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations


def getOptimalParameterForSVM_alpha_multiprocess(train_tfidf, train_data, train_target, target_performance,
                                                 candidate_alpha, process_number, queue):
    saveDatasetIndices = True
    # print "Process " + str(process_number) + "started"
    class_weight = 'balanced'
    optimal_alpha = -1
    max_performance_measure = -1
    standard_deviation = -1
    all_measures_performances = []
    all_measures_standardDeviations = []

    fiveFoldModels = []
    fiveFoldTrainingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTestingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTrainingLabels = []  # save the 5 fold training labels
    fiveFoldTestingLabels = []  # Save the 5 fold testing labels

    # Do 5-folds cross validation on training data to determine the best C value
    skf = StratifiedKFold(n_splits=5, random_state=9876, shuffle=True)
    target_performances_of_all_folds = []

    # Store ALL the average performance measures (Accuracy, AUC, macro precision, macro recall, macro F1) to display in the GUI
    performances_all_measures_all_folds = []
    standard_deviations_all_measures_all_folds = []

    fiveFoldModelsCandidate = []

    for train_indices, test_indices in skf.split(train_tfidf, train_target):
        # Note: It is okay to use tf-idf transformed data for doing CV, because tf-idf is unsupervised
        X_train, X_test = train_tfidf[train_indices], train_tfidf[test_indices]
        y_train, y_test = np.array(train_target)[train_indices], np.array(train_target)[test_indices]

        X_train_original, X_test_original = [train_data[i] for i in train_indices], [train_data[j] for j in
                                                                                     test_indices]

        random_state = np.random.RandomState(seed=3456)

        # When using SGD, the partial_fit method has to be applied on different batches of the training data, and we need to epoch multiple times
        model_fold = SGDClassifier(loss='hinge', penalty='l2', alpha=candidate_alpha, random_state=random_state)

        def batches(l, n):
            for i in np.arange(0, len(l), n):
                yield l[i:i + n]

        n_iter = 25
        np.random.seed(5647)
        shuffledRange = np.arange(len(X_train))
        for n in np.arange(n_iter):
            np.random.shuffle(shuffledRange)
            shuffled_X_train = [X_train[i] for i in shuffledRange]
            shuffled_y_train = [y_train[i] for i in shuffledRange]

            # Training the model in 10 batches
            for batch in batches(np.arange(len(shuffled_y_train)), 5):
                model_fold.partial_fit(shuffled_X_train[batch[0]:batch[-1] + 1],
                                       shuffled_y_train[batch[0]:batch[-1] + 1], classes=np.unique(y_train))

        fiveFoldModelsCandidate.append(model_fold)

        (accu, auc, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1,
         pred_y) = evaluate_model_MS(model_fold, X_test, y_test, list(set(y_train)))

        if target_performance == 'accuracy':
            target_performances_of_all_folds.append(accu)
        elif target_performance == 'macro_f1':
            target_performances_of_all_folds.append(macro_f1)
        elif target_performance == 'macro_precision':
            target_performances_of_all_folds.append(macro_precision)
        elif target_performance == 'macro_recall':
            target_performances_of_all_folds.append(macro_recall)
        elif target_performance == 'auc':
            target_performances_of_all_folds.append(auc)

        performances_all_measures_one_fold = []
        performances_all_measures_one_fold.append(accu)
        performances_all_measures_one_fold.append(auc)
        performances_all_measures_one_fold.append(macro_precision)
        performances_all_measures_one_fold.append(macro_recall)
        performances_all_measures_one_fold.append(macro_f1)

        performances_all_measures_all_folds.append(performances_all_measures_one_fold)

        # Update progressbar to show progress of each fold; increment the progressbar value by 1 each time in this for loop
        # progress_value += 1

    saveDatasetIndices = False

    # Take an average of target_performances_of_all_folds
    if max_performance_measure <= np.average(target_performances_of_all_folds):
        max_performance_measure = np.average(target_performances_of_all_folds)
        standard_deviation = np.std(target_performances_of_all_folds)
        optimal_alpha = candidate_alpha

        # Additionally, store the average performance and standard deviations of all measures
        all_measures_performances = np.average(performances_all_measures_all_folds, axis=0)
        all_measures_standardDeviations = np.std(performances_all_measures_all_folds, axis=0)

        fiveFoldModels = fiveFoldModelsCandidate

    final_res = (optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
                 all_measures_standardDeviations)
    queue.put([optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
               all_measures_standardDeviations])
    return


def getOptimalParameterForSVM_alpha(train_tfidf, train_data, train_target, target_performance):
    """
    Function to compute optimal parameter, alpha, for MNB model
    """
    all_candidates_alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    processes = []
    queue = mp.Queue()
    results = np.ndarray(len(all_candidates_alpha), dtype=object)
    for i in range(len(all_candidates_alpha)):
        arguments = train_tfidf, train_data, train_target, target_performance, all_candidates_alpha[i], i, queue
        p = Process(target=getOptimalParameterForSVM_alpha_multiprocess, args=arguments, )
        p.daemon = True
        processes.append(p)
        processes[i].start()
    for i in range(len(all_candidates_alpha)):
        processes[i].join()
    for i in range(len(all_candidates_alpha)):
        results[i] = queue.get()
        # Compare which candidate alpha gives the maximum performance
    max_performances = []
    for result in results:
        max_performances.append(result[1])

    optimal_alpha_index = np.argmax(max_performances)

    final_result = results[optimal_alpha_index]
    optimal_alpha = final_result[0]
    max_performance_measure = final_result[1]
    standard_deviation = final_result[2]
    all_measures_performances = final_result[3]
    all_measures_standardDeviations = final_result[4]

    return optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations


def getOptimalParameterForOVRMNB_alpha_multiprocess(train_tfidf, train_data, train_target, target_performance,
                                                    candidate_alpha, process_number, queue):
    saveDatasetIndices = True
    # print "Process " + str(process_number) + "started"
    class_weight = 'balanced'
    optimal_alpha = -1
    max_performance_measure = -1
    standard_deviation = -1
    all_measures_performances = []
    all_measures_standardDeviations = []

    fiveFoldModels = []
    fiveFoldTrainingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTestingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTrainingLabels = []  # save the 5 fold training labels
    fiveFoldTestingLabels = []  # Save the 5 fold testing labels

    # Do 5-folds cross validation on training data to determine the best C value
    skf = StratifiedKFold(n_splits=5, random_state=9876, shuffle=True)
    target_performances_of_all_folds = []

    # Store ALL the average performance measures (Accuracy, AUC, macro precision, macro recall, macro F1) to display in the GUI
    performances_all_measures_all_folds = []
    standard_deviations_all_measures_all_folds = []

    fiveFoldModelsCandidate = []

    for train_indices, test_indices in skf.split(train_tfidf, train_target):
        # Note: It is okay to use tf-idf transformed data for doing CV, because tf-idf is unsupervised
        X_train, X_test = train_tfidf[train_indices], train_tfidf[test_indices]
        y_train, y_test = np.array(train_target)[train_indices], np.array(train_target)[test_indices]

        X_train_original, X_test_original = [train_data[i] for i in train_indices], [train_data[j] for j in
                                                                                     test_indices]

        model_fold = OneVsRestClassifier(MultinomialNB(alpha=candidate_alpha)).partial_fit(X_train, y_train,
                                                                                           classes=np.unique(y_train))

        fiveFoldModelsCandidate.append(model_fold)

        (accu, auc, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1,
         pred_y) = evaluate_model_MS(model_fold, X_test, y_test, list(set(y_train)))

        if target_performance == 'accuracy':
            target_performances_of_all_folds.append(accu)
        elif target_performance == 'macro_f1':
            target_performances_of_all_folds.append(macro_f1)
        elif target_performance == 'macro_precision':
            target_performances_of_all_folds.append(macro_precision)
        elif target_performance == 'macro_recall':
            target_performances_of_all_folds.append(macro_recall)
        elif target_performance == 'auc':
            target_performances_of_all_folds.append(auc)

        performances_all_measures_one_fold = []
        performances_all_measures_one_fold.append(accu)
        performances_all_measures_one_fold.append(auc)
        performances_all_measures_one_fold.append(macro_precision)
        performances_all_measures_one_fold.append(macro_recall)
        performances_all_measures_one_fold.append(macro_f1)

        performances_all_measures_all_folds.append(performances_all_measures_one_fold)

        # Update progressbar to show progress of each fold; increment the progressbar value by 1 each time in this for loop
        # progress_value += 1

    saveDatasetIndices = False

    # Take an average of target_performances_of_all_folds
    if max_performance_measure <= np.average(target_performances_of_all_folds):
        max_performance_measure = np.average(target_performances_of_all_folds)
        standard_deviation = np.std(target_performances_of_all_folds)
        optimal_alpha = candidate_alpha

        # Additionally, store the average performance and standard deviations of all measures
        all_measures_performances = np.average(performances_all_measures_all_folds, axis=0)
        all_measures_standardDeviations = np.std(performances_all_measures_all_folds, axis=0)

        fiveFoldModels = fiveFoldModelsCandidate

    final_res = (optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
                 all_measures_standardDeviations)
    queue.put([optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
               all_measures_standardDeviations])
    return


def getOptimalParameterForOVRMNB_alpha(train_tfidf, train_data, train_target, target_performance):
    """
    Function to compute optimal parameter, alpha, for MNB model
    """
    all_candidates_alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    processes = []
    queue = mp.Queue()
    results = np.ndarray(len(all_candidates_alpha), dtype=object)
    for i in range(len(all_candidates_alpha)):
        arguments = train_tfidf, train_data, train_target, target_performance, all_candidates_alpha[i], i, queue
        p = Process(target=getOptimalParameterForOVRMNB_alpha_multiprocess, args=arguments, )
        p.daemon = True
        processes.append(p)
        processes[i].start()
    for i in range(len(all_candidates_alpha)):
        processes[i].join()
    for i in range(len(all_candidates_alpha)):
        results[i] = queue.get()
        # Compare which candidate alpha gives the maximum performance
    max_performances = []
    for result in results:
        max_performances.append(result[1])

    optimal_alpha_index = np.argmax(max_performances)

    final_result = results[optimal_alpha_index]
    optimal_alpha = final_result[0]
    max_performance_measure = final_result[1]
    standard_deviation = final_result[2]
    all_measures_performances = final_result[3]
    all_measures_standardDeviations = final_result[4]

    return optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations


def getOptimalParameterForOVRLR_alpha_multiprocess(train_tfidf, train_data, train_target, target_performance,
                                                   candidate_alpha, process_number, queue):
    saveDatasetIndices = True
    # print "Process " + str(process_number) + "started"
    class_weight = 'balanced'
    optimal_alpha = -1
    max_performance_measure = -1
    standard_deviation = -1
    all_measures_performances = []
    all_measures_standardDeviations = []

    fiveFoldModels = []
    fiveFoldTrainingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTestingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTrainingLabels = []  # save the 5 fold training labels
    fiveFoldTestingLabels = []  # Save the 5 fold testing labels

    # Do 5-folds cross validation on training data to determine the best C value
    skf = StratifiedKFold(n_splits=5, random_state=9876, shuffle=True)
    target_performances_of_all_folds = []

    # Store ALL the average performance measures (Accuracy, AUC, macro precision, macro recall, macro F1) to display in the GUI
    performances_all_measures_all_folds = []
    standard_deviations_all_measures_all_folds = []

    fiveFoldModelsCandidate = []

    for train_indices, test_indices in skf.split(train_tfidf, train_target):
        # Note: It is okay to use tf-idf transformed data for doing CV, because tf-idf is unsupervised
        X_train, X_test = train_tfidf[train_indices], train_tfidf[test_indices]
        y_train, y_test = np.array(train_target)[train_indices], np.array(train_target)[test_indices]

        X_train_original, X_test_original = [train_data[i] for i in train_indices], [train_data[j] for j in
                                                                                     test_indices]

        random_state = np.random.RandomState(seed=3456)

        # When using SGD, the partial_fit method has to be applied on different batches of the training data, and we need to epoch multiple times
        model_fold = OneVsRestClassifier(
            SGDClassifier(loss='log', penalty='l2', alpha=candidate_alpha, random_state=random_state))

        def batches(l, n):
            for i in np.arange(0, len(l), n):
                yield l[i:i + n]

        n_iter = 25
        np.random.seed(5647)
        shuffledRange = np.arange(len(X_train))
        for n in np.arange(n_iter):
            np.random.shuffle(shuffledRange)
            shuffled_X_train = [X_train[i] for i in shuffledRange]
            shuffled_y_train = [y_train[i] for i in shuffledRange]

            # Training the model in 10 batches
            for batch in batches(np.arange(len(shuffled_y_train)), 5):
                model_fold.partial_fit(shuffled_X_train[batch[0]:batch[-1] + 1],
                                       shuffled_y_train[batch[0]:batch[-1] + 1], classes=np.unique(y_train))

        fiveFoldModelsCandidate.append(model_fold)

        (accu, auc, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1,
         pred_y) = evaluate_model_MS(model_fold, X_test, y_test, list(set(y_train)))

        if target_performance == 'accuracy':
            target_performances_of_all_folds.append(accu)
        elif target_performance == 'macro_f1':
            target_performances_of_all_folds.append(macro_f1)
        elif target_performance == 'macro_precision':
            target_performances_of_all_folds.append(macro_precision)
        elif target_performance == 'macro_recall':
            target_performances_of_all_folds.append(macro_recall)
        elif target_performance == 'auc':
            target_performances_of_all_folds.append(auc)

        performances_all_measures_one_fold = []
        performances_all_measures_one_fold.append(accu)
        performances_all_measures_one_fold.append(auc)
        performances_all_measures_one_fold.append(macro_precision)
        performances_all_measures_one_fold.append(macro_recall)
        performances_all_measures_one_fold.append(macro_f1)

        performances_all_measures_all_folds.append(performances_all_measures_one_fold)

        # Update progressbar to show progress of each fold; increment the progressbar value by 1 each time in this for loop
        # progress_value += 1

    saveDatasetIndices = False

    # Take an average of target_performances_of_all_folds
    if max_performance_measure <= np.average(target_performances_of_all_folds):
        max_performance_measure = np.average(target_performances_of_all_folds)
        standard_deviation = np.std(target_performances_of_all_folds)
        optimal_alpha = candidate_alpha

        # Additionally, store the average performance and standard deviations of all measures
        all_measures_performances = np.average(performances_all_measures_all_folds, axis=0)
        all_measures_standardDeviations = np.std(performances_all_measures_all_folds, axis=0)

        fiveFoldModels = fiveFoldModelsCandidate

    final_res = (optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
                 all_measures_standardDeviations)
    queue.put([optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
               all_measures_standardDeviations])
    return


def getOptimalParameterForOVRLR_alpha(train_tfidf, train_data, train_target, target_performance):
    """
    Function to compute optimal parameter, alpha, for MNB model
    """
    all_candidates_alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    processes = []
    queue = mp.Queue()
    results = np.ndarray(len(all_candidates_alpha), dtype=object)
    for i in range(len(all_candidates_alpha)):
        arguments = train_tfidf, train_data, train_target, target_performance, all_candidates_alpha[i], i, queue
        p = Process(target=getOptimalParameterForOVRLR_alpha_multiprocess, args=arguments, )
        p.daemon = True
        processes.append(p)
        processes[i].start()
    for i in range(len(all_candidates_alpha)):
        processes[i].join()
    for i in range(len(all_candidates_alpha)):
        results[i] = queue.get()
        # Compare which candidate alpha gives the maximum performance
    max_performances = []
    for result in results:
        max_performances.append(result[1])

    optimal_alpha_index = np.argmax(max_performances)

    final_result = results[optimal_alpha_index]
    optimal_alpha = final_result[0]
    max_performance_measure = final_result[1]
    standard_deviation = final_result[2]
    all_measures_performances = final_result[3]
    all_measures_standardDeviations = final_result[4]

    return optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations


def getOptimalParameterForOVRSVM_alpha_multiprocess(train_tfidf, train_data, train_target, target_performance,
                                                    candidate_alpha, process_number, queue):
    saveDatasetIndices = True
    # print "Process " + str(process_number) + "started"
    class_weight = 'balanced'
    optimal_alpha = -1
    max_performance_measure = -1
    standard_deviation = -1
    all_measures_performances = []
    all_measures_standardDeviations = []

    fiveFoldModels = []
    fiveFoldTrainingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTestingDatasetTfidf = []  # datasets remain the same for all candidate_alphas, only the model changes
    fiveFoldTrainingLabels = []  # save the 5 fold training labels
    fiveFoldTestingLabels = []  # Save the 5 fold testing labels

    # Do 5-folds cross validation on training data to determine the best C value
    skf = StratifiedKFold(n_splits=5, random_state=9876, shuffle=True)
    target_performances_of_all_folds = []

    # Store ALL the average performance measures (Accuracy, AUC, macro precision, macro recall, macro F1) to display in the GUI
    performances_all_measures_all_folds = []
    standard_deviations_all_measures_all_folds = []

    fiveFoldModelsCandidate = []

    for train_indices, test_indices in skf.split(train_tfidf, train_target):
        # Note: It is okay to use tf-idf transformed data for doing CV, because tf-idf is unsupervised
        X_train, X_test = train_tfidf[train_indices], train_tfidf[test_indices]
        y_train, y_test = np.array(train_target)[train_indices], np.array(train_target)[test_indices]

        X_train_original, X_test_original = [train_data[i] for i in train_indices], [train_data[j] for j in
                                                                                     test_indices]

        random_state = np.random.RandomState(seed=3456)

        # When using SGD, the partial_fit method has to be applied on different batches of the training data, and we need to epoch multiple times
        model_fold = OneVsRestClassifier(
            SGDClassifier(loss='hinge', penalty='l2', alpha=candidate_alpha, random_state=random_state))

        def batches(l, n):
            for i in np.arange(0, len(l), n):
                yield l[i:i + n]

        n_iter = 25
        np.random.seed(5647)
        shuffledRange = np.arange(len(X_train))
        for n in np.arange(n_iter):
            np.random.shuffle(shuffledRange)
            shuffled_X_train = [X_train[i] for i in shuffledRange]
            shuffled_y_train = [y_train[i] for i in shuffledRange]

            # Training the model in 10 batches
            for batch in batches(np.arange(len(shuffled_y_train)), 5):
                model_fold.partial_fit(shuffled_X_train[batch[0]:batch[-1] + 1],
                                       shuffled_y_train[batch[0]:batch[-1] + 1], classes=np.unique(y_train))

        # Save the indices for only one candidate_alpha; indices do not change with different candidate_alphas!
        if saveDatasetIndices == True:
            fiveFoldTrainingDatasetTfidf.append(X_train_original)
            fiveFoldTestingDatasetTfidf.append(X_test_original)
            fiveFoldTrainingLabels.append(y_train)
            fiveFoldTestingLabels.append(y_test)

        fiveFoldModelsCandidate.append(model_fold)

        (accu, auc, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1,
         pred_y) = evaluate_model_MS(model_fold, X_test, y_test, list(set(y_train)))

        if target_performance == 'accuracy':
            target_performances_of_all_folds.append(accu)
        elif target_performance == 'macro_f1':
            target_performances_of_all_folds.append(macro_f1)
        elif target_performance == 'macro_precision':
            target_performances_of_all_folds.append(macro_precision)
        elif target_performance == 'macro_recall':
            target_performances_of_all_folds.append(macro_recall)
        elif target_performance == 'auc':
            target_performances_of_all_folds.append(auc)

        performances_all_measures_one_fold = []
        performances_all_measures_one_fold.append(accu)
        performances_all_measures_one_fold.append(auc)
        performances_all_measures_one_fold.append(macro_precision)
        performances_all_measures_one_fold.append(macro_recall)
        performances_all_measures_one_fold.append(macro_f1)

        performances_all_measures_all_folds.append(performances_all_measures_one_fold)

        # Update progressbar to show progress of each fold; increment the progressbar value by 1 each time in this for loop
        # progress_value += 1

    saveDatasetIndices = False

    # Take an average of target_performances_of_all_folds
    if max_performance_measure <= np.average(target_performances_of_all_folds):
        max_performance_measure = np.average(target_performances_of_all_folds)
        standard_deviation = np.std(target_performances_of_all_folds)
        optimal_alpha = candidate_alpha

        # Additionally, store the average performance and standard deviations of all measures
        all_measures_performances = np.average(performances_all_measures_all_folds, axis=0)
        all_measures_standardDeviations = np.std(performances_all_measures_all_folds, axis=0)

        fiveFoldModels = fiveFoldModelsCandidate

    final_res = (optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
                 all_measures_standardDeviations)
    queue.put([optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances,
               all_measures_standardDeviations])
    return


def getOptimalParameterForOVRSVM_alpha(train_tfidf, train_data, train_target, target_performance):
    """
    Function to compute optimal parameter, alpha, for MNB model
    """
    all_candidates_alpha = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    processes = []
    queue = mp.Queue()
    results = np.ndarray(len(all_candidates_alpha), dtype=object)
    for i in range(len(all_candidates_alpha)):
        arguments = train_tfidf, train_data, train_target, target_performance, all_candidates_alpha[i], i, queue
        p = Process(target=getOptimalParameterForOVRSVM_alpha_multiprocess, args=arguments, )
        p.daemon = True
        processes.append(p)
        processes[i].start()
    for i in range(len(all_candidates_alpha)):
        processes[i].join()
    for i in range(len(all_candidates_alpha)):
        results[i] = queue.get()
        # Compare which candidate alpha gives the maximum performance
    max_performances = []
    for result in results:
        max_performances.append(result[1])

    optimal_alpha_index = np.argmax(max_performances)

    final_result = results[optimal_alpha_index]
    optimal_alpha = final_result[0]
    max_performance_measure = final_result[1]
    standard_deviation = final_result[2]
    all_measures_performances = final_result[3]
    all_measures_standardDeviations = final_result[4]

    return optimal_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations

def getOptimalParametersForDeepLearning(dl_model_type, dl_model_name, train_data, target_performance, output_dir):
    """
    Function to compute optimal parameters, batch size and max_sequence_length, for deep learning models
    We will not perform CV, but instead do a train/test split, since 5-fold CV is too expensive for deep learning!
    """
    all_candidates_batchsize = [16, 32]
    all_candidates_maxSequenceLength = [256, 512]
    max_performance_measure = -1
    # Optimal parameters contains two values: first value is optimal batch size and second value is optimal max sequence length
    optimal_parameters = []
    train_data = train_data.split('\n')
    train_data_header = train_data[0]
    train_data = train_data[1:]
    print(0)
    for candidate_batchsize in all_candidates_batchsize:
        print(1)
        for candidate_max_sequence_length in all_candidates_maxSequenceLength:
            print(2)
            train, test = train_test_split(train_data, test_size=0.2, random_state=4937, shuffle=True)
            train_str = ''
            for t in train:
                train_str += t + '\n'
            train = train_data_header + '\n' + train_str
            print(3)
            # !rm - r
            # outputs
            # Create a TransformerModel
            trainedModel = DeepLearningModel(dl_model_type, dl_model_name, candidate_max_sequence_length,
                                             candidate_batchsize, num_epochs=30, random_state=4987,
                                             output_dir=output_dir)
            print(4)
            trainedModel.fit(train)
            print(5)
            test_str = ''
            for t1 in test:
                test_str += t1 + '\n'
            test = train_data_header + '\n' + test_str
            (accu, auc, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1,
             pred_y) = evaluate_model_MS(trainedModel, test, None, None)
            print(6)
            if target_performance == 'accuracy':
                target_performance_value = accu
            elif target_performance == 'macro_f1':
                target_performance_value = macro_f1
            elif target_performance == 'macro_precision':
                target_performance_value = macro_precision
            elif target_performance == 'macro_recall':
                target_performance_value = macro_recall
            elif target_performance == 'auc':
                target_performance_value = auc
            print(7)
            if max_performance_measure < target_performance_value:
                optimal_parameters = []
                # First parameter in this list is batch size, and second parameter is max sequence length
                optimal_parameters.append(candidate_batchsize)
                optimal_parameters.append(candidate_max_sequence_length)
                max_performance_measure = target_performance_value

    return optimal_parameters, max_performance_measure, "N/A", None, None


def getBestModelAndHyperParameters(train_tfidf, train_data, train_target, target_performance):
    """
    Function to compute the best model AND the best hyper-parameters related to each model
    """
    # listOfModels = ['mnb', 'lrl2', 'svm', 'ovrmnb', 'ovrlrl2', 'ovrsvm']
    # Excluded svm and one-vs-rest svm from the automatic model selection procedure, because these two models take a very long time for training
    listOfModels = ['mnb', 'lrl2', 'ovrmnb', 'ovrlrl2']

    ret_chosen_model = ''
    ret_optimal_model_parameter = -1
    best_max_performance_measure = -1
    ret_max_performance_measure = None
    ret_standard_deviation = None
    ret_all_measures_performances = None
    ret_all_measures_standardDeviations = None
    ret_fiveFoldTrainingDatasetTfidf = None
    ret_fiveFoldTestingDatasetTfidf = None
    ret_fiveFoldModels = None
    fiveFoldTrainingDatasetTfidf = None
    fiveFoldTestingDatasetTfidf = None
    fiveFoldTrainingLabels = None
    fiveFoldTestingLabels = None
    fiveFoldModels = None

    for candidate_model in listOfModels:
        # For each model, compute the performance and store it in array; this will be used to compare the models

        if candidate_model == 'mnb':
            # Get optimal alpha for the model
            progress_text = progress_text + "\nEvaluating Multinomial Naive Bayes model..."

            mnb_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations = getOptimalParameterForMNB_alpha(
                train_tfidf.todense(), train_data, train_target, target_performance)

            if max_performance_measure >= best_max_performance_measure:
                ret_chosen_model = 'Multinomial Naive Bayes'
                ret_optimal_model_parameter = mnb_alpha
                best_max_performance_measure = max_performance_measure
                ret_max_performance_measure = max_performance_measure
                ret_standard_deviation = standard_deviation
                ret_all_measures_performances = all_measures_performances
                ret_all_measures_standardDeviations = all_measures_standardDeviations

                ret_fiveFoldTrainingDatasetTfidf = fiveFoldTrainingDatasetTfidf
                ret_fiveFoldTestingDatasetTfidf = fiveFoldTestingDatasetTfidf
                ret_fiveFoldModels = fiveFoldTestingDatasetTfidf

            progress_text += "\nModel parameter optimization complete."
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nResults for 5-fold cross validation on training data"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nOptimal smoothing parameter (alpha) value: " + str(mnb_alpha)
            progress_text += "\nBest target " + ": " + str(
                np.round(max_performance_measure * 100.0, 2)) + "% +/- " + str(
                np.round(standard_deviation * 100.0, 2)) + "%"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage model performance across all measures using the optimal parameter:"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage Accuracy: " + str(
                np.round(all_measures_performances[0] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[0] * 100.0, 2)) + "%"
            progress_text += "\nAverage AUC: " + str(
                np.round(all_measures_performances[1] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[1] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Precision: " + str(
                np.round(all_measures_performances[2] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[2] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Recall: " + str(
                np.round(all_measures_performances[3] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[3] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro F1: " + str(
                np.round(all_measures_performances[4] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[4] * 100.0, 2)) + "%"
            progress_text += "\n***************************************************************************"


        elif candidate_model == 'lrl2':
            # Get optimal C for the model
            progress_text += "\n***************************************************************************"
            progress_text += "\nEvaluating Logistic Regression model..."

            lrl2_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations = getOptimalParameterForLR_alpha(
                train_tfidf.todense(), train_data, train_target, target_performance)

            if max_performance_measure >= best_max_performance_measure:
                ret_chosen_model = 'Logistic Regression'
                ret_optimal_model_parameter = lrl2_alpha
                best_max_performance_measure = max_performance_measure
                ret_max_performance_measure = max_performance_measure
                ret_standard_deviation = standard_deviation
                ret_all_measures_performances = all_measures_performances
                ret_all_measures_standardDeviations = all_measures_standardDeviations

                ret_fiveFoldTrainingDatasetTfidf = fiveFoldTrainingDatasetTfidf
                ret_fiveFoldTestingDatasetTfidf = fiveFoldTestingDatasetTfidf
                ret_fiveFoldModels = fiveFoldTestingDatasetTfidf

            progress_text += "\nModel parameter optimization complete."
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nResults for 5-fold cross validation on training data"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nOptimal regularization parameter (C) value: " + str(lrl2_alpha)
            progress_text += "\nBest target " + ": " + str(
                np.round(max_performance_measure * 100.0, 2)) + "% +/- " + str(
                np.round(standard_deviation * 100.0, 2)) + "%"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage model performance across all measures using the optimal parameter:"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage Accuracy: " + str(
                np.round(all_measures_performances[0] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[0] * 100.0, 2)) + "%"
            progress_text += "\nAverage AUC: " + str(
                np.round(all_measures_performances[1] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[1] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Precision: " + str(
                np.round(all_measures_performances[2] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[2] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Recall: " + str(
                np.round(all_measures_performances[3] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[3] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro F1: " + str(
                np.round(all_measures_performances[4] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[4] * 100.0, 2)) + "%"
            progress_text += "\n***************************************************************************"


        elif candidate_model == 'svm':
            progress_text += "\n***************************************************************************"
            progress_text += "\nEvaluating Support Vector Machines model..."

            # Get optimal C for the model, performance of 5-fold CV, and standard deviation of performance
            svm_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations = getOptimalParameterForSVM_alpha(
                train_tfidf.todense(), train_data, train_target, target_performance)

            if max_performance_measure >= best_max_performance_measure:
                ret_chosen_model = 'Support Vector Machines'
                ret_optimal_model_parameter = svm_alpha
                best_max_performance_measure = max_performance_measure
                ret_max_performance_measure = max_performance_measure
                ret_standard_deviation = standard_deviation
                ret_all_measures_performances = all_measures_performances
                ret_all_measures_standardDeviations = all_measures_standardDeviations

                ret_fiveFoldTrainingDatasetTfidf = fiveFoldTrainingDatasetTfidf
                ret_fiveFoldTestingDatasetTfidf = fiveFoldTestingDatasetTfidf
                ret_fiveFoldModels = fiveFoldTestingDatasetTfidf

            progress_text += "\nModel parameter optimization complete."
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nResults for 5-fold cross validation on training data"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\Optimal penalty parameter of the error term (C) value: " + str(svm_alpha)
            progress_text += "\nBest target " + ": " + str(
                np.round(max_performance_measure * 100.0, 2)) + "% +/- " + str(
                np.round(standard_deviation * 100.0, 2)) + "%"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage model performance across all measures using the optimal parameter:"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage Accuracy: " + str(
                np.round(all_measures_performances[0] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[0] * 100.0, 2)) + "%"
            progress_text += "\nAverage AUC: " + str(
                np.round(all_measures_performances[1] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[1] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Precision: " + str(
                np.round(all_measures_performances[2] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[2] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Recall: " + str(
                np.round(all_measures_performances[3] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[3] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro F1: " + str(
                np.round(all_measures_performances[4] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[4] * 100.0, 2)) + "%"
            progress_text += "\n***************************************************************************"

        if candidate_model == 'ovrmnb':
            # Get optimal alpha for the model
            progress_text += "\n***************************************************************************"
            progress_text += "\nEvaluating One vs Rest (Multinomial Naive Bayes) model..."

            mnb_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations = getOptimalParameterForOVRMNB_alpha(
                train_tfidf.todense(), train_data, train_target, target_performance)

            if max_performance_measure >= best_max_performance_measure:
                ret_chosen_model = 'One vs Rest (Multinomial Naive Bayes)'
                ret_optimal_model_parameter = mnb_alpha
                best_max_performance_measure = max_performance_measure
                ret_max_performance_measure = max_performance_measure
                ret_standard_deviation = standard_deviation
                ret_all_measures_performances = all_measures_performances
                ret_all_measures_standardDeviations = all_measures_standardDeviations

                ret_fiveFoldTrainingDatasetTfidf = fiveFoldTrainingDatasetTfidf
                ret_fiveFoldTestingDatasetTfidf = fiveFoldTestingDatasetTfidf
                ret_fiveFoldModels = fiveFoldTestingDatasetTfidf

            progress_text += "\nModel parameter optimization complete."
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nResults for 5-fold cross validation on training data"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nOptimal smoothing parameter (alpha) value: " + str(mnb_alpha)
            progress_text += "\nBest target " + ": " + str(
                np.round(max_performance_measure * 100.0, 2)) + "% +/- " + str(
                np.round(standard_deviation * 100.0, 2)) + "%"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage model performance across all measures using the optimal parameter:"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage Accuracy: " + str(
                np.round(all_measures_performances[0] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[0] * 100.0, 2)) + "%"
            progress_text += "\nAverage AUC: " + str(
                np.round(all_measures_performances[1] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[1] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Precision: " + str(
                np.round(all_measures_performances[2] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[2] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Recall: " + str(
                np.round(all_measures_performances[3] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[3] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro F1: " + str(
                np.round(all_measures_performances[4] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[4] * 100.0, 2)) + "%"
            progress_text += "\n***************************************************************************"


        elif candidate_model == 'ovrlrl2':
            # Get optimal C for the model
            progress_text += "\n***************************************************************************"
            progress_text += "\nEvaluating One vs Rest (Logistic Regression) model..."

            lrl2_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations = getOptimalParameterForOVRLR_alpha(
                train_tfidf.todense(), train_data, train_target, target_performance)

            if max_performance_measure >= best_max_performance_measure:
                ret_chosen_model = 'One vs Rest (Logistic Regression)'
                ret_optimal_model_parameter = lrl2_alpha
                best_max_performance_measure = max_performance_measure
                ret_max_performance_measure = max_performance_measure
                ret_standard_deviation = standard_deviation
                ret_all_measures_performances = all_measures_performances
                ret_all_measures_standardDeviations = all_measures_standardDeviations

                ret_fiveFoldTrainingDatasetTfidf = fiveFoldTrainingDatasetTfidf
                ret_fiveFoldTestingDatasetTfidf = fiveFoldTestingDatasetTfidf
                ret_fiveFoldModels = fiveFoldTestingDatasetTfidf

            progress_text += "\nModel parameter optimization complete."
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nResults for 5-fold cross validation on training data"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nOptimal regularization parameter (C) value: " + str(lrl2_alpha)
            progress_text += "\nBest target " + str(
                np.round(max_performance_measure * 100.0, 2)) + "% +/- " + str(
                np.round(standard_deviation * 100.0, 2)) + "%"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage model performance across all measures using the optimal parameter:"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage Accuracy: " + str(
                np.round(all_measures_performances[0] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[0] * 100.0, 2)) + "%"
            progress_text += "\nAverage AUC: " + str(
                np.round(all_measures_performances[1] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[1] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Precision: " + str(
                np.round(all_measures_performances[2] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[2] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Recall: " + str(
                np.round(all_measures_performances[3] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[3] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro F1: " + str(
                np.round(all_measures_performances[4] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[4] * 100.0, 2)) + "%"
            progress_text += "\n***************************************************************************"


        elif candidate_model == 'ovrsvm':
            progress_text += "\n***************************************************************************"
            progress_text += "\nEvaluating One vs Rest (Support Vector Machines) model..."

            # Get optimal C for the model, performance of 5-fold CV, and standard deviation of performance
            svm_alpha, max_performance_measure, standard_deviation, all_measures_performances, all_measures_standardDeviations = getOptimalParameterForOVRSVM_alpha(
                train_tfidf.todense(), train_data, train_target, target_performance)

            if max_performance_measure >= best_max_performance_measure:
                ret_chosen_model = 'One vs Rest (Support Vector Machines)'
                ret_optimal_model_parameter = svm_alpha
                best_max_performance_measure = max_performance_measure
                ret_max_performance_measure = max_performance_measure
                ret_standard_deviation = standard_deviation
                ret_all_measures_performances = all_measures_performances
                ret_all_measures_standardDeviations = all_measures_standardDeviations

                ret_fiveFoldTrainingDatasetTfidf = fiveFoldTrainingDatasetTfidf
                ret_fiveFoldTestingDatasetTfidf = fiveFoldTestingDatasetTfidf
                ret_fiveFoldModels = fiveFoldTestingDatasetTfidf

            progress_text += "\nModel parameter optimization complete."
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nResults for 5-fold cross validation on training data"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\Optimal penalty parameter of the error term (C) value: " + str(svm_alpha)
            progress_text += "\nBest target " + ": " + str(
                np.round(max_performance_measure * 100.0, 2)) + "% +/- " + str(
                np.round(standard_deviation * 100.0, 2)) + "%"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage model performance across all measures using the optimal parameter:"
            progress_text += "\n---------------------------------------------------------------------------"
            progress_text += "\nAverage Accuracy: " + str(
                np.round(all_measures_performances[0] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[0] * 100.0, 2)) + "%"
            progress_text += "\nAverage AUC: " + str(
                np.round(all_measures_performances[1] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[1] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Precision: " + str(
                np.round(all_measures_performances[2] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[2] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro Recall: " + str(
                np.round(all_measures_performances[3] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[3] * 100.0, 2)) + "%"
            progress_text += "\nAverage Macro F1: " + str(
                np.round(all_measures_performances[4] * 100.0, 2)) + "% +/- " + str(
                np.round(all_measures_standardDeviations[4] * 100.0, 2)) + "%"
            progress_text += "\n***************************************************************************"

    return ret_chosen_model, ret_optimal_model_parameter, ret_max_performance_measure, ret_standard_deviation, ret_all_measures_performances, ret_all_measures_standardDeviations


def evaluate_model_MS(model, X_test, y_test, y_label_set):
    # initialize all peformance measures to -1 (i.e., not defined)
    accu = -1
    auc = -1
    micro_precision = -1
    macro_precision = -1
    micro_recall = -1
    macro_recall = -1
    micro_f1 = -1
    macro_f1 = -1
    # Note: The y_test needs to be a binary array of [n_samples, n_classes] where each value indicates presence/absence of class label in the respective column
    if not isinstance(model, DeepLearningModel):
        labelBinarizer = preprocessing.LabelBinarizer()
        labelBinarizer.fit(y_label_set)
        binarizedLabels = labelBinarizer.transform(y_test)

    if isinstance(model, MultinomialNB) or isinstance(model, KNeighborsClassifier) or isinstance(model,
                                                                                                 DeepLearningModel):
        # KNN Classifier's predict_proba function does not work with sparse matrices
        if isinstance(model, KNeighborsClassifier):
            y_probas = model.predict_proba(X_test)
        elif isinstance(model, DeepLearningModel):
            print('Start deep learning prediction')
            predict_proba_output = model.predict_proba(X_test)
            y_probas  = predict_proba_output['predicted_probabilities']
            y_test = predict_proba_output['test_list_y']
            print('Finish deep learning prediction')
            print(y_test)
            labelBinarizer = preprocessing.LabelBinarizer()
            y_label_set = set(y_test)
            labelBinarizer.fit(y_label_set)
            binarizedLabels = labelBinarizer.transform(y_test)
        else:
            y_probas = model.predict_proba(X_test)
        try:
            auc = metrics.roc_auc_score(binarizedLabels, y_probas)
        except:
            # If AUC cannot be computed, set AUC value to -1, to represent not defined
            auc = -1
    elif isinstance(model, OneVsRestClassifier) and isinstance(model.estimator, MultinomialNB):
        y_probas = model.predict_proba(X_test)
        try:
            auc = metrics.roc_auc_score(binarizedLabels, y_probas)
        except:
            # If AUC cannot be computed, set AUC value to -1, to represent not defined
            auc = -1
    else:
        # For LR and SVM:
        y_decision = model.decision_function(X_test)
        try:
            auc = metrics.roc_auc_score(binarizedLabels, y_decision)
        except:
            # If AUC cannot be computed, set AUC value to -1, to represent not defined
            auc = -1

    if isinstance(model, DeepLearningModel):
        print('Evaluating deep learning')
        pred_y, y_test = model.predict(X_test)
        print('Finish evaluating deep learning')
    else:
        pred_y = model.predict(X_test)
    accu = metrics.accuracy_score(y_test, pred_y)
    micro_precision = metrics.precision_score(y_test, pred_y, average='micro')
    macro_precision = metrics.precision_score(y_test, pred_y, average='macro')
    micro_recall = metrics.recall_score(y_test, pred_y, average='micro')
    macro_recall = metrics.recall_score(y_test, pred_y, average='macro')
    micro_f1 = metrics.f1_score(y_test, pred_y, average='micro')
    macro_f1 = metrics.f1_score(y_test, pred_y, average='macro')

    print(accu,micro_precision)
    print(macro_precision)
    return (accu, auc, micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1, pred_y)
