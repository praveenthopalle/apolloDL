# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import multiprocessing
import random
from threading import Thread

import botocore
from django.contrib import auth
from django.contrib.auth import authenticate
from django.shortcuts import render
from django.template import RequestContext
from django.utils.datetime_safe import datetime
from django.views.decorators.csrf import csrf_exempt
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer as wn
from django.http import HttpResponse, JsonResponse
import re
from openpyxl import load_workbook
from openpyxl.writer.excel import save_virtual_workbook

from apollo.Lib.collections import Counter
from apollo4.ComputeOptimalParameters import getOptimalParameterForMNB_alpha, getOptimalParameterForLR_alpha, \
    getOptimalParameterForSVM_alpha, getOptimalParameterForOVRMNB_alpha, getOptimalParameterForOVRLR_alpha, \
    getOptimalParameterForOVRSVM_alpha, getBestModelAndHyperParameters
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import glob
import copy
from dateutil.relativedelta import relativedelta
import os, sys, signal
from os.path import splitext
import uuid
import apollo4.globals
import boto3
from elasticsearch import Elasticsearch
from apollo4.connection import MyConnection
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from zipfile import ZipFile
import io
from django.contrib.auth.decorators import login_required
from django.conf import settings
from array import array
import copy

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd",
             "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers",
             "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
             "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been",
             "being",
             "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
             "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
             "through",
             "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
             "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
             "any", "both",
             "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
             "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d",
             "ll", "m",
             "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't",
             "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't",
             "mustn", "mustn't",
             "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
             "won't", "wouldn", "wouldn't"]


@csrf_exempt
@login_required(login_url='/login/')
def sl(request):
    return render(request, 'apollo4/spotlight.html')

def Spotlight_Top_Papers(df):
    try:
        # if request.method == 'POST':
        #     userName = request.user.username;
        #     inputData = request.FILES.getlist('file')
        #     inputDataArray = []
        #     altogether = None
        #     titleRecords = []
        # for files in inputData:
        #     inputrecords = pd.read_csv(files, header=0, encoding='unicode_escape')
        #     title = inputrecords.get('Title')
        #     titleRecords.append(title)
        #     inputDataArray.append(inputrecords)
        #
        # print(titleRecords)
        # df = pd.read_csv()
        # df.head(5)
        # # Get topk from GUI (e.g., top 5, top 10, top 15, or top 20)

        topk = 10
        num_references = df['Cited by']
        titles = df['Title']
        years = df['Year']
        authors = df['Authors']
        affiliations = df['Affiliations']
        topkindices = np.argsort(list(num_references))[::-1]
        topkindices_cleaned = []

        counter = 0
        while len(topkindices_cleaned) < topk:
            if not np.isnan(num_references[topkindices[counter]]):
                topkindices_cleaned.append(topkindices[counter])
            counter += 1

        num_references[topkindices_cleaned]

        # authors_without_commas = authors[topkindices_cleaned]
        #
        # authors_to_return = []
        # for author in authors_without_commas:
        #     authors_to_return.append(author.replace(',', ';'))
        topPapers = {
            'titles': titles[topkindices_cleaned].to_json(),
            'authors': authors[topkindices_cleaned].to_json(),
            'affiliations':affiliations[topkindices_cleaned].to_json(),
            'years':years[topkindices_cleaned].to_json(),
            'citations': num_references[topkindices_cleaned].to_json()
        }
        # return titles, years, authors, and affiliations for top papers
        return topPapers
    except Exception as e:
        return HttpResponse(
            "Error running the program. Please contact the IP Group Analytics Team (apolloip@ssi.samsung.com) to resolve the issue. Please provide the error details below in your email. \nPlease provide all the steps to reproduce this issue. \n" + "-" * 40 + "\n" + str(
                e) + "\n" + "-" * 40)


def Spotlight_Top_Patents(df):
    try:
        # df = pd.read_csv(datafile, sep=',', encoding='ISO-8859-1')
        # print(df.head(5))
        # Get topk from GUI (e.g., top 5, top 10, top 15, or top 20)
        topk = 10
        references = df['Domestic References-By']
        num_references = []
        for ref in references:
            if type(ref) == str:
                if ref != '':
                    num_ref = len(ref.split(','))
                    num_references.append(num_ref)
                else:
                    num_references.append(0)
            elif type(ref) == float:
                num_references.append(0)
            else:
                num_references.append(0)

        titles = df['Title']
        application_dates = df['Appl. Date']
        assignees = df['Assignee']
        topkindices = np.argsort(num_references)[::-1]
        topkindices_cleaned = []
        num_references = pd.DataFrame(num_references, columns=['References'])
        num_references = num_references['References']
        topPatents = {
            'titles': titles[topkindices[:topk]].to_json(),
            'assignees': assignees[topkindices[:topk]].to_json(),
            'application_dates': application_dates[topkindices[:topk]].to_json(),
            'citations': num_references[topkindices[:topk]].to_json()
        }
        # return titles, years, authors, and affiliations for top papers
        return topPatents
    except Exception as e:
        return HttpResponse(
            "Error running the program. Please contact the IP Group Analytics Team (apolloip@ssi.samsung.com) to resolve the issue. Please provide the error details below in your email. \nPlease provide all the steps to reproduce this issue. \n" + "-" * 40 + "\n" + str(
                e) + "\n" + "-" * 40)


def Spotlight_Top_Institutions_From_Patents(df):
    # Get topk from GUI (e.g., top 5, top 10, top 15, or top 20)
    try:
        topk = 10

        dict_assignees = Counter(df['Assignee'])

        unique_assignees = []
        patent_count_per_assignee = []
        for assignee in dict_assignees:
            if type(assignee) != str:
                continue
            else:
                unique_assignees.append(assignee)
                patent_count_per_assignee.append(dict_assignees[assignee])

        top_indices = np.argsort(patent_count_per_assignee)[::-1][:topk]

        top_assignees = []
        top_count_patents = []
        for idx in top_indices:
            top_assignees.append(unique_assignees[idx])
            top_count_patents.append(patent_count_per_assignee[idx])

        topPatentsInstitutions = {
            'top_assignees': top_assignees,
            'top_count_patents': top_count_patents,
        }
        return topPatentsInstitutions
    except Exception as e:
        return HttpResponse(
            "Error running the program. Please contact the IP Group Analytics Team (apolloip@ssi.samsung.com) to resolve the issue. Please provide the error details below in your email. \nPlease provide all the steps to reproduce this issue. \n" + "-" * 40 + "\n" + str(
                e) + "\n" + "-" * 40)


@csrf_exempt
# Assuming that the function received filesList in csv format.
def Spotlight_Process_All_Files(request):
    try:
        if request.method == 'POST':
            userName = request.user.username;
            inputData = request.FILES.getlist('file')
            allPatentData = pd.DataFrame()
            allJournalData = pd.DataFrame()
            typePatentOrJournal = []
            # One or multiple files may have issues. Create an error string to display to the user in case of errors.
            errorString = ''
            header = None

            counter = 0
            for file in inputData:
                print(file)
                test_file = file
                # SIPMS original downloaded data might contain four extra lines before the header. So check if the first four lines contain SIPMS specific information.
                line1 = test_file.readline().decode("utf-8")

                if 'nasca' in line1.lower():
                    errorString += 'File "' + file + '" is NASCA encrypted. Please decrypt the file and try again.'
                    continue

                line2 = test_file.readline().decode("utf-8")
                line3 = test_file.readline().decode("utf-8")
                test_file.seek(0)
                datatype = None

                rows_to_skip = 0
                if 'Created date' in line1 and 'Database' in line2 and 'Search expression' in line3:
                    # Based on this, assume that this data came from SIPMS, and remove the first four lines from the data
                    datatype = 'Patent'
                    rows_to_skip = 4

                # Check if the file contains patent or journal data
                if datatype == 'Patent':
                    df = pd.read_csv(file, sep=',', encoding='ISO-8859-1', skiprows=rows_to_skip)
                    # line5 = file.readline().decode("utf-8")
                    # line6 = file.readline().decode("utf-8")
                    # line6 = line6.replace('\ufeff', '')
                    # line6 = line6.replace('\n', '')
                    # line6 = line6.split(',')
                    # header = line6
                else:
                    df = pd.read_csv(file)
                    # line1 = line1.replace('\ufeff', '')
                    # line1 = line1.replace('\n', '')
                    # line1 = line1.split(',')
                    #header = line1
                header = list(df.head())

                counter += 1

                # Check the headers to confirm file type as patent or journal.
                list_of_all_columns_in_SIPMS = ['No.', 'Title', 'Abstract', 'Appl. No.', 'Appl. Date', 'Pub. No.',
                                                'Pub. Date',
                                                'Pub. No. (Exam.)', 'Pub. Date (Exam.)', 'Reg. No.', 'Reg. Date',
                                                'Assignee',
                                                'Assignee Address', 'Assignee English Name', 'Assignee Country',
                                                'Normalized Assignee', 'Current Assignee', 'Inventors',
                                                'Inventor Country',
                                                'Priority No.', 'PCT Application No.', 'PCT Application Date',
                                                'PCT No.',
                                                'PCT Publication Date', 'Family', 'Examiner', 'Attorney',
                                                'Designated Countries',
                                                'Independent claims count', 'Claims Count', 'First Claim',
                                                'First Claim(Native)',
                                                'Claims', 'Claims(Native)', 'Full-text HyperLink', 'Title(Native)',
                                                'Abstract(Native)', 'IPC', 'CPC', 'UPC', 'Domestic References',
                                                'Foreign References', 'Domestic References-By', 'Domestic Ref. Count',
                                                'Domestic Ref. By Count', 'Life Status', 'Final Status',
                                                'Estimated Termination Date', 'Patent Evaluation Grade',
                                                'Patent Evaluation Score',
                                                'Inventor Evaluation Grade', 'Inventor Evaluation Score',
                                                'Inventor Count',
                                                'Transfer of Ownership', 'Assignment Date', 'Inpadoc Family Count',
                                                'Inpadoc Family Country', 'Inpadoc Family Country Count',
                                                'Genealogy Count',
                                                'Internal Reference', 'Country ID', 'GBM Name',
                                                'Management Affiliation Code',
                                                'Samsung Product Classification', 'Samsung Technology Classification']
                list_of_all_columns_in_SCOPUS = ['Authors', 'Author(s) ID', 'Title', 'Year', 'Source title', 'Volume',
                                                 'Issue',
                                                 'Art. No.', 'Page start', 'Page end', 'Page count', 'Cited by', 'DOI',
                                                 'Link',
                                                 'Affiliations', 'Authors with affiliations', 'Abstract',
                                                 'Author Keywords',
                                                 'Index Keywords', 'Molecular Sequence Numbers', 'Chemicals/CAS',
                                                 'Tradenames',
                                                 'Manufacturers', 'Funding Details', 'Funding Text 1', 'Funding Text 2',
                                                 'References', 'Correspondence Address', 'Editors', 'Sponsors',
                                                 'Publisher',
                                                 'Conference name', 'Conference date', 'Conference location',
                                                 'Conference code',
                                                 'ISSN', 'ISBN', 'CODEN', 'PubMed ID', 'Language of Original Document',
                                                 'Abbreviated Source Title', 'Document Type', 'Publication Stage',
                                                 'Access Type',
                                                 'Source', 'EID']

                if set(header).issubset(set(list_of_all_columns_in_SCOPUS)):
                    datatype = 'Journal'
                elif set(header).issubset(set(list_of_all_columns_in_SIPMS)):
                    datatype = 'Patent'

                if datatype == 'Patent':
                    # Check whether necessary information is available in the data. If not, prepare error string to display to the user.
                    if 'Title' not in header:
                        errorString += 'The file ' + file + ' does not contain "Title" column.\n'
                    if 'Appl. No.' not in header:
                        errorString += 'The file ' + file + ' does not contain "Appl. No." column.\n'
                    if 'Appl. Date' not in header:
                        errorString += 'The file ' + file + ' does not contain "Appl. Date." column.\n'
                    if 'Assignee' not in header:
                        errorString += 'The file ' + file + ' does not contain "Assignee" column.\n'
                    if 'Domestic References-By' not in header:
                        errorString += 'The file ' + file + ' does not contain "Domestic References-By" column.\n'

                elif datatype == 'Journal':
                    # Check whether necessary information is available in the data. If not, prepare error string to display to the user.
                    if 'Title' not in header:
                        errorString += 'The file ' + file + ' does not contain "Title" column.\n'
                    if 'Year' not in header:
                        errorString += 'The file ' + file + ' does not contain "Year" column.\n'
                    if 'Authors' not in header:
                        errorString += 'The file ' + file + ' does not contain "Authors" column.\n'
                    if 'Affiliations' not in header:
                        errorString += 'The file ' + file + ' does not contain "Affiliations" column.\n'
                    if 'Cited by' not in header:
                        errorString += 'The file ' + file + ' does not contain "Cited By" column.\n'

                else:
                    errorString += 'Cannot extract necessary information from the file ' + file + '.\n'

                # Combine data based on whether it is patent or journal, and only if the file had no errors
                if errorString == '':
                    if datatype == 'Patent':
                        # Combine data to patents
                        if len(allPatentData) == 0:
                            # Create a new dataframe
                            allPatentData = df.filter(
                                ['Title', 'Appl. No.', 'Appl. Date', 'Assignee', 'Domestic References-By'],
                                axis=1)
                        else:
                            # Merge patent data to existing dataframe
                            new_list = df.filter(
                                ['Title', 'Appl. No.', 'Appl. Date', 'Assignee', 'Domestic References-By'], axis=1)
                            pd_lists = [allPatentData,
                                        new_list[
                                            ['Title', 'Appl. No.', 'Appl. Date', 'Assignee', 'Domestic References-By']]]
                            allPatentData = pd.concat(pd_lists, ignore_index=True)

                    elif datatype == 'Journal':
                        # Combine data to journals
                        if len(allJournalData) == 0:
                            allJournalData = df.filter(['Title', 'Year', 'Authors', 'Affiliations', 'Cited by'], axis=1)
                        else:
                            # Merge journal data to existing dataframe
                            new_list = df.filter(['Title', 'Year', 'Authors', 'Affiliations', 'Cited by'], axis=1)
                            pd_lists = [allJournalData,
                                        new_list[['Title', 'Year', 'Authors', 'Affiliations', 'Cited by']]]
                            allJournalData = pd.concat(pd_lists, ignore_index=True)
                    else:
                        # If any other error happened and we were unable to read the file
                        errorString += 'Cannot read the file "' + file + '".\n'

                else:
                    errorString += 'Cannot extract necessary information from the file "' + file + '".\n'

            if errorString == '':
                topPaperTable = Spotlight_Top_Papers(allJournalData)
                topPatentTable = Spotlight_Top_Patents(allPatentData)
                topInstitutionsPatents = Spotlight_Top_Institutions_From_Patents(allPatentData)
                table = {
                    'topPaperTable':topPaperTable,
                    'topPatentTable':topPatentTable,
                    'topInstitutionsPatents':topInstitutionsPatents
                }
                result = table
                errorString = False
            else:
                result = errorString
                errorString = True

        return JsonResponse({'result': result, 'errorString': errorString})
    except Exception as e:
        return HttpResponse(
            "Error running the program. Please contact the IP Group Analytics Team (apolloip@ssi.samsung.com) to resolve the issue. Please provide the error details below in your email. \nPlease provide all the steps to reproduce this issue. \n" + "-" * 40 + "\n" + str(
                e) + "\n" + "-" * 40)


# Assuming that the function received filesList in csv or txt format.
def Process_All_Files(filesList):
    allPatentData = []
    allJournalData = []
    allData = None

    # Category tracks whether the files contains Category column, which is necessary for training/ labeled data.
    categories_provided = []

    # One or multiple files may have issues. Create an error string to display to the user in case of errors.
    errorString = ''

    # Check the headers to confirm file type as patent or journal. Included Category column in the list of headers for parsing training/labeled data.
    list_of_all_columns_in_SIPMS = ['No.', 'Title', 'Abstract', 'Appl. No.', 'Appl. Date', 'Pub. No.', 'Pub. Date',
                                    'Pub. No. (Exam.)', 'Pub. Date (Exam.)', 'Reg. No.', 'Reg. Date', 'Assignee',
                                    'Assignee Address', 'Assignee English Name', 'Assignee Country',
                                    'Normalized Assignee', 'Current Assignee', 'Inventors', 'Inventor Country',
                                    'Priority No.', 'PCT Application No.', 'PCT Application Date', 'PCT No.',
                                    'PCT Publication Date', 'Family', 'Examiner', 'Attorney', 'Designated Countries',
                                    'Independent claims count', 'Claims Count', 'First Claim', 'First Claim(Native)',
                                    'Claims', 'Claims(Native)', 'Full-text HyperLink', 'Title(Native)',
                                    'Abstract(Native)', 'IPC', 'CPC', 'UPC', 'Domestic References',
                                    'Foreign References', 'Domestic References-By', 'Domestic Ref. Count',
                                    'Domestic Ref. By Count', 'Life Status', 'Final Status',
                                    'Estimated Termination Date', 'Patent Evaluation Grade', 'Patent Evaluation Score',
                                    'Inventor Evaluation Grade', 'Inventor Evaluation Score', 'Inventor Count',
                                    'Transfer of Ownership', 'Assignment Date', 'Inpadoc Family Count',
                                    'Inpadoc Family Country', 'Inpadoc Family Country Count', 'Genealogy Count',
                                    'Internal Reference', 'Country ID', 'GBM Name', 'Management Affiliation Code',
                                    'Samsung Product Classification', 'Samsung Technology Classification', 'Category']
    list_of_all_columns_in_SCOPUS = ['Authors', 'Author(s) ID', 'Title', 'Year', 'Source title', 'Volume', 'Issue',
                                     'Art. No.', 'Page start', 'Page end', 'Page count', 'Cited by', 'DOI', 'Link',
                                     'Affiliations', 'Authors with affiliations', 'Abstract', 'Author Keywords',
                                     'Index Keywords', 'Molecular Sequence Numbers', 'Chemicals/CAS', 'Tradenames',
                                     'Manufacturers', 'Funding Details', 'Funding Text 1', 'Funding Text 2',
                                     'References', 'Correspondence Address', 'Editors', 'Sponsors', 'Publisher',
                                     'Conference name', 'Conference date', 'Conference location', 'Conference code',
                                     'ISSN', 'ISBN', 'CODEN', 'PubMed ID', 'Language of Original Document',
                                     'Abbreviated Source Title', 'Document Type', 'Publication Stage', 'Access Type',
                                     'Source', 'EID', 'Category']
    list_of_all_columns_in_Old_Patent_Specification = ['Identification Number', 'Title', 'Abstract', 'Claims',
                                                       'Application Number', 'Application Date', 'Current Assignee',
                                                       'Upc', 'Category']
    list_of_all_columns_in_Old_Paper_Specification = ['Meta Data', 'Title', 'Abstract', 'Author', 'Affiliation',
                                                      'Published Year', 'Category']

    counter = 0
    for file in filesList:
        print(file)
        test_file = file
        # SIPMS original downloaded data might contain four extra lines before the header. So check if the first four lines contain SIPMS specific information.
        line1 = test_file.readline().decode("ISO-8859-1")

        if 'nasca' in line1.lower():
            errorString += 'File "' + file + '" is NASCA encrypted. Please decrypt the file and try again.'
            continue

        line2 = test_file.readline().decode("ISO-8859-1")
        line3 = test_file.readline().decode("ISO-8859-1")
        test_file.seek(0)
        datatype = None

        rows_to_skip = 0
        if 'Created date' in line1 and 'Database' in line2 and 'Search expression' in line3:
            # Based on this, assume that this data came from SIPMS, and remove the first four lines from the data
            datatype = 'Patent'
            rows_to_skip = 4
            fileformat = 'new'

        # The function should still support old file formats.
        elif 'identification number\ttitle\tabstract\tclaims\tapplication number\tapplication date\tcurrent assignee\tupc' in line1.lower():
            # it is patent data
            datatype = 'Patent'
            fileformat = 'old'
            if 'category' in line1.lower():
                category = True
            else:
                category = False

        elif 'meta data\ttitle\tabstract\tauthor\taffiliation\tpublished year' in line1.lower():
            # it is journal data
            datatype = 'Journal'
            fileformat = 'old'
            if 'category' in line1.lower():
                category = True
            else:
                category = False

        if fileformat == 'old':
            data = file.read().decode("ISO-8859-1")
            # Concatenate data based on patent or journal
            if datatype == 'Patent':
                if len(allPatentData) == 0:
                    allPatentData = data
                else:
                    if allPatentData.endswith('\n'):
                        allPatentData += data
                    else:
                        allPatentData += '\n' + data.split('\n')[1]
                allData = allPatentData

            elif datatype == 'Journal':
                if len(allJournalData) == 0:
                    allJournalData = data
                else:
                    if allJournalData.endswith('\n'):
                        allJournalData += data
                    else:
                        allJournalData += '\n' + data.split('\n')[1]
                allData = allJournalData

        else:
            # The file may be in new format from SCOPUS or SIPMS, or may be unreadable.
            # Check if the file contains patent or journal data

            if datatype == 'Patent' and fileformat == 'new':
                try:
                    df = pd.read_csv(file, sep=',', encoding='ISO-8859-1', skiprows=rows_to_skip)
                except:
                    errorString += 'Cannot open file "' + file + '".\n'
                    categories_provided.append(category)
                    continue
            else:
                # Maybe this is journal data. Check the header to confirm whether this file contains journal data.
                try:
                    df = pd.read_csv(file)
                except:
                    try:
                        # Maybe this file is in ISO-8859-1 format, so try parsing the file using this encoding
                        df = pd.read_csv(file, sep=',', encoding='ISO-8859-1', skiprows=0)
                    except:
                        errorString += 'Cannot open file "' + file + '".\n'
                        categories_provided.append(category)
                        continue

            header = list(df.head())

            if header != None:
                if set(header).issubset(set(list_of_all_columns_in_SCOPUS)):
                    datatype = 'Journal'

                elif set(header).issubset(set(list_of_all_columns_in_SIPMS)):
                    datatype = 'Patent'
                elif set(header).issubset(set(list_of_all_columns_in_Old_Paper_Specification)):
                    datatype = 'Journal'
                    fileformat = 'old'

                elif set(header).issubset(set(list_of_all_columns_in_Old_Patent_Specification)):
                    datatype = 'Patent'
                    fileformat = 'old'

                if 'category' in set(header):
                    category = True
                else:
                    category = False

            if datatype == 'Patent':
                identification_number_header = ''
                application_number_header = ''
                application_date_header = ''

                if 'Identification Number' in header:
                    identification_number_header = 'Identification Number'
                elif 'No.' in header:
                    identification_number_header = 'No.'
                else:
                    identification_number_header = 'Identification Number'
                    # Add a new column to dataframe
                    df.insert(0, identification_number_header, 'unknown')

                if 'Appl. No.' in header:
                    application_number_header = 'Appl. No.'
                elif 'Application Number' in header:
                    application_number_header = 'Application Number'
                else:
                    application_number_header = 'Missing'

                if 'Appl. Date' in header:
                    application_date_header = 'Appl. Date'
                elif 'Application Date' in header:
                    application_date_header = 'Application Date'
                else:
                    application_date_header = 'Missing'

                if 'Title' not in header:
                    errorString += 'The file ' + file + ' does not contain "Title" column.\n'
                if 'Abstract' not in header:
                    errorString += 'The file ' + file + ' does not contain "Abstract" column.\n'
                if 'Claims' not in header:
                    errorString += 'The file ' + file + ' does not contain "Claims" column.\n'
                if 'Appl. No.' not in header:
                    errorString += 'The file ' + file + ' does not contain "Appl. No." column.\n'
                if 'Appl. Date' not in header:
                    errorString += 'The file ' + file + ' does not contain "Appl. Date" column.\n'
                if 'Assignee' not in header:
                    errorString += 'The file ' + file + ' does not contain "Assignee" column.\n'
                if 'UPC' not in header:
                    errorString += 'The file ' + file + ' does not contain "UPC" column.\n'

            elif datatype == 'Journal':
                # Check whether necessary information is available in the data. If not, prepare error string to display to the user.
                identification_number_header = ''
                if 'Meta Data' in header:
                    identification_number_header = 'Meta Data'
                else:
                    identification_number_header = 'Meta Data'
                    df.insert(0, identification_number_header, 'unknown')

                if 'Title' not in header:
                    errorString += 'The file ' + file + ' does not contain "Title" column.\n'
                if 'Abstract' not in header:
                    errorString += 'The file ' + file + ' does not contain "Abstract" column.\n'

                if fileformat == 'old' and 'Author' not in header:
                    errorString += 'The file ' + file + ' does not contain "Author" column.\n'
                if fileformat == 'old' and 'Affiliation' not in header:
                    errorString += 'The file ' + file + ' does not contain "Affiliation" column.\n'

                if fileformat != 'old' and 'Authors' not in header:
                    errorString += 'The file ' + file + ' does not contain "Authors" column.\n'
                if fileformat != 'old' and 'Affiliations' not in header:
                    errorString += 'The file ' + file + ' does not contain "Affiliations" column.\n'

                if fileformat == 'old' and 'Published Year' not in header:
                    errorString += 'The file ' + file + ' does not contain "Published Year" column.\n'
                if fileformat != 'old' and 'Year' not in header:
                    errorString += 'The file ' + file + ' does not contain "Year" column.\n'

            else:
                errorString += 'Cannot extract necessary information from the file "' + file + '".\n'

            # Combine data based on whether it is patent or journal, and only if the file had no errors while processing
            if errorString == '':
                if datatype == 'Patent':
                    # Combine data with patents data

                    if category == True:
                        patentData = df.filter(
                            [identification_number_header, 'Title', 'Abstract', 'Claims', application_number_header,
                             application_date_header, 'Assignee', 'UPC', 'Category'], axis=1)
                    else:
                        patentData = df.filter(
                            [identification_number_header, 'Title', 'Abstract', 'Claims', application_number_header,
                             application_date_header, 'Assignee', 'UPC'], axis=1)

                    # Convert pandas dataframe to tab delimited file and merge into allPatentData
                    if len(allPatentData) == 0:
                        patentData_tab_delimited = patentData.to_csv(index=False, header=True, sep='\t')
                        allPatentData = patentData_tab_delimited
                    else:
                        patentData_tab_delimited = patentData.to_csv(index=False, header=False, sep='\t')
                        if allPatentData.endswith('\n'):
                            allPatentData += patentData_tab_delimited
                        else:
                            allPatentData += '\n' + patentData_tab_delimited
                    allData = allPatentData

                elif datatype == 'Journal':
                    # Combine data with journals data

                    if category == True:
                        if fileformat == 'old':
                            journalData = df.filter(
                                [identification_number_header, 'Title', 'Abstract', 'Author', 'Affiliation', 'Year',
                                 'Category'], axis=1)
                        else:
                            journalData = df.filter(
                                [identification_number_header, 'Title', 'Abstract', 'Authors', 'Affiliations', 'Year',
                                 'Category'], axis=1)

                    else:
                        if fileformat == 'old':
                            journalData = df.filter(
                                [identification_number_header, 'Title', 'Abstract', 'Author', 'Affiliation', 'Year'],
                                axis=1)
                        else:
                            journalData = df.filter(
                                [identification_number_header, 'Title', 'Abstract', 'Authors', 'Affiliations', 'Year'],
                                axis=1)

                    if len(allJournalData) == 0:
                        journalData_tab_delimited = journalData.to_csv(index=False, header=True, sep='\t')
                        allJournalData = journalData_tab_delimited
                    else:
                        journalData_tab_delimited = journalData.to_csv(index=False, header=False, sep='\t')
                        if allJournalData.endswith('\n'):
                            allJournalData += journalData_tab_delimited
                        else:
                            allJournalData += '\n' + journalData_tab_delimited
                    allData = allJournalData
                else:
                    # If any other error happened and we were unable to read the file
                    errorString += 'Cannot read the file "' + file + '".\n'

            else:
                errorString += 'Cannot extract necessary information from the file "' + file + '".\n'

        categories_provided.append(category)

    return allData
#
#
# # Example call to the function
# datafiles = ['Sample_Paper_Data.csv', 'Sample_Patent_Data_2.csv']
#
# # Other example calls to the function with multiple patent and journal files.
# # datafiles = ['Sample_Paper_Data.csv', 'Sample_Patent_Data_2.csv', 'Sample_Paper_Data.csv', 'Sample_Patent_Data_2.csv']
# datafiles = ['Patent_old_format.txt', 'Journal_old_format.txt']
# datafiles = ['Sample_Paper_Data.csv', 'Sample_Paper_Data.csv', 'Sample_Patent_Data_2.csv', 'Sample_Patent_Data_2.csv',
#              'Journal_old_format.txt', 'Patent_old_format.txt']
#
# allJournalData, allPatentData, errorString = Process_All_Files(datafiles)
# print(allJournalData)
# print(allPatentData)
# print(len(allJournalData.split('\n')))
# print(len(allPatentData.split('\n')))
# print(errorString)































