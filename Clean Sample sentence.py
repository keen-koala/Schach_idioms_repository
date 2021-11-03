# -*- coding: utf-8 -*-

from csv import reader, writer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import csv
import re
from removeaccents import removeaccents
import unicodedata
import os
import time
from SortedSet.sorted_set import SortedSet


startTime = time.time()

'''This function creates a sample of 100 sentences per idiom_id, if less than 100 it takes the shape (len lines of of the idiom_id group) '''
def get_sample(dataframe, sample_size):
	try:
		sample_dataframe = dataframe.sample(n = sample_size, random_state=1)
	except ValueError:
		sample_size = dataframe.shape[0]
		sample_dataframe = dataframe.sample(n= sample_size, random_state =1)
	return sample_dataframe


filename = "C:/Users/Vanessa/Desktop/Masterarbeit/MA_repository/respository/idiom_sentences_final.tsv"
csv_input = pd.read_csv(filename, sep="\t", header=0)
all_idioms = csv_input["idiom_id"].to_list() # all_idioms = [1,1,2,2,2]
unique_ids = SortedSet(all_idioms) # a set is a datastructure which contains only unique items --> unique_ids = set(1,2)

dfs = []  # empty list which will hold dataframes

for id in unique_ids: # for every group with a unique id, you want to draw a sample of size x
    grouped_data= csv_input[csv_input['idiom_id']==id] # create an extra dataframe which only consists of the items where the 'idiom_id' equals the id
    group_sample = get_sample(grouped_data, 100)
    dfs.append(group_sample)
all_dataframes = pd.concat(dfs)  # concatenate list of dataframes
all_dataframes.reset_index(drop = True).to_csv('sentiment_random_100_lines_exception_2209.tsv', sep='\t', index= False)#writes data in tsv with original writing



'''This  function inserts the cleaned sentences '''
def cleaned_context_frames(dataframe):
	#Iterate through random sample and replace sentence breaks and accents
	#sentence_list=[]
	dataframe["sentence"]=''
	dataframe.reset_index(drop=True, inplace=True)
	for reihe, row in enumerate(dataframe.itertuples(index=False,name=None)):  # [['Left','KWIC','Right']] # itertuples is faster in runtime than iterrow, reihe is index
		print(reihe)
		index = str(row[0])
		left = str(row[6])
		#print(left)
		kwic= str(row[7])
		#print( "idiom: ", kwic)
		right= str(row[8])
		#print(right)
		complete_sentence= left + ' ' + kwic + ' ' + right
		#print("sentence: ", complete_sentence)
		split_sentence=complete_sentence.split("</s><s>")
		#print("split sentence: ", index, split_sentence)

		for sentence in split_sentence:
			if kwic in sentence:
				#print("This is the cleaned sentence: ", sentence)
				dataframe.at[reihe,"sentence"]=sentence  # insert sentence in row
				#return kwic
		#sentence_list.append(sentence)


		#print(len(sentence_list))

	#all_sentences = pd.concat(sentence_list)  # concatenate list of dataframes
	#dataframe= dataframe.drop(['Left','KWIC','Right'], axis=1)
	#dataframe.insert(6, 'sentence', sentence_list)  # insert idiom id
	# dataframe['sentence'] = [sentence for sentence in sentence_list]# add sentence to new dataframe
	return dataframe


#cleaned_context_frames(all_dataframes)
result_dataframe=cleaned_context_frames(csv_input).reset_index(drop=True).to_csv('sentences_full_corpus_cleaned_201021.tsv', sep='\t',
												index=False)  # writes data in tsv with cleaned sentences (cleaned_context_frames(all_dataframes))
print("all done")


#def annotate_sentiment(annotator, sample):

# filename = "C:/Users/Vanessa/Desktop/Masterarbeit/MA_repository/respository/idiom_sentences_final.tsv"
# delete_character_lits=["</s><s>", "'", ]
# sentiment_annotator_list= ["Cardiff_NLP", "NLP_town", "Vader"]
# Cardiff_NLP_list=[]
# NLP_town_list=[]
# Vader_list=[]
#
# '''Load Models'''
# ''' Pipeline with CardiffNLP'''
# model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# sentiment_task_cardiff = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
#
# ''' NLP town Model '''  # Instantiate a pipeline object with our task and model passed as parameters,  #https://www.kdnuggets.com/2021/06/create-deploy-sentiment-analysis-app-api.html
# sentiment_task_nlptown = pipeline(task='sentiment-analysis',
#                                   model='nlptown/bert-base-multilingual-uncased-sentiment')
#
#
# csv_input = pd.read_csv(filename, sep="\t", header=0)
#
#
#
# '''Create a random sample of 100 per idiom_id. If fewer then one 100 leave number as it is '''
# #https://blog.softhints.com/pandas-random-sample-of-a-subset-of-a-dataframe-rows-or-columns/
# csv_random_sample_100=csv_input.groupby('idiom_id').apply(lambda x: x.sample(n=100, random_state=1, replace=False)).reset_index(drop = True)# 100 sentences per idiom_id
# #csv_random_sample_100 = csv_input.sample(n=100)
#
#
# ''' Iterate through random sample and replace sentence breaks and accents '''
# for row in csv_random_sample_100.itertuples(index=False, name=None):#[['Left','KWIC','Right']] # itertuples is faster in runtime than iterrow
# #for row in csv_input.itertuples(index=False, name=None):#[['Left','KWIC','Right']] # itertuples is faster in runtime than iterrow
#     sentence=str(row[6:9])# Left, KWIC, Right
#     index= str(row[0])
#     print(index)
#     print(sentence)
#
#     char_to_replace = {'</s><s>': '',
#                              "''": "'"
#                              }#https://thispointer.com/python-replace-multiple-characters-in-a-string/
#     # Iterate over all key-value pairs in dictionary
#     for key, value in char_to_replace.items():
#         #     # Replace key character with value character in string
#         sentence = sentence.replace(key, value)
#         sentence_lower = str(sentence).lower()  # convert to lowercase for processing in sentiment annotator
#         sentence_no_accents = removeaccents.remove_accents(sentence_lower)
#     print(index, ':', sentence_no_accents)
#
#     ''' Pipeline with CardiffNLP'''
#     cardiff_sentiment_label = (sentiment_task_cardiff(sentence_no_accents)[0]['label'])# access label
#
# # here the label in Cardiff is converted to binary number
#     if cardiff_sentiment_label== 'Neutral':
#         cardiff_sentiment_label= '0.0'
#         Cardiff_NLP_list.append(cardiff_sentiment_label)
#     elif cardiff_sentiment_label== 'Negative':
#         cardiff_sentiment_label = '-1.0'
#         Cardiff_NLP_list.append(cardiff_sentiment_label)
#     else:
#         cardiff_sentiment_label = '1.0'
#         Cardiff_NLP_list.append(cardiff_sentiment_label)
#
#     print('Cardiff NLP: ', str(cardiff_sentiment_label))
#
#
#     ''' NLP town Model '''  # Instantiate a pipeline object with our task and model passed as parameters,  #https://www.kdnuggets.com/2021/06/create-deploy-sentiment-analysis-app-api.html
#     nlp_town_sentiment_label = sentiment_task_nlptown(sentence_no_accents)[0]['label'] # access label in result
#     nlp_town_sentiment_label= re.sub(r'([a-z]+)',"", nlp_town_sentiment_label)#replace('star')
#     #print(nlp_town_sentiment_label)
#     NLP_town_list.append(nlp_town_sentiment_label)
#
#     # Pass the text to our pipeline and print the results
#     print('NLP town: ', str(nlp_town_sentiment_label))
#
#     ''' Using Vader'''  # https://github.com/cjhutto/vaderSentiment
#     # using vader multihttps://pypi.org/project/vader-multi/
#     # https://github.com/brunneis/vader-multi
#
#     analyzer = SentimentIntensityAnalyzer()
#     vader_sentiment_label = analyzer.polarity_scores(sentence_no_accents)['compound']# accesses compund score
#     Vader_list.append(vader_sentiment_label)
#     print('Vader: ', vader_sentiment_label)
#
# print(NLP_town_list)
# print(Cardiff_NLP_list)
# print(Vader_list)
#
#
# # add new columns in dataframe with sentiment, then the are saved in new csv
# csv_random_sample_100['NLP_town_stars'] = [label for label in NLP_town_list]
# csv_random_sample_100['Cardiff_NLP'] = [label for label in Cardiff_NLP_list]
# csv_random_sample_100['Vader'] = [label for label in Vader_list]
# csv_random_sample_100.to_csv('sentiment_random_100_lines_1709.tsv', sep='\t')#writes data in tsv with original writing
#
#
#
# executionTime = (time.time() - startTime)
# print('Execution time in seconds: ' + str(executionTime))
# #Execution time in seconds: 1072.0286486148834