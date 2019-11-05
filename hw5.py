"""This is a sample file for hw5. 
It contains the functions that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

import json
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math

model = None

"""
– “label” – A Boolean value saying whether this is a sarcastic comment
– “comment” – the text of the comment itself
– “author” – the username of the comment’s author
– “subreddit” – the name of the subreddit this comment came from
– “ups” / “downs” – the # of upvotes / downvotes this comment got
– ”parent_comment” – the text of the comment this one replied to
"""

def trainSarcasm(Model, trainFile):
	global model
	model = Model
	#TODO: process training file here
	#load in the json list file into a list of dictionaries
	jsonlist = list()
	f = open(trainFile, "r")
	for line in f:
		jsonlist.append(json.loads(line))
	
	#convert words to vectors
	vectorlist = list()

	for d in jsonlist:
		w_vec_list = list()
		for w in d["comment"]:
			vector = model[w]
			w_vec_list.append(vector)
		#comment vector - mean of word vectors
		s_vec = sum(w_vec_list) / len(w_vec_list)
		#parent comment vector

		#
		

	model = keras.Sequential([
		keras.layers.Flatten(input_shape=(28, 28)),
		keras.layers.Dense(128, activation='relu'),
		keras.layers.Dense(2, activation='softmax')	# prob of true and false
	])
	

def testSarcasm(comment):
	return random.choice([True,False])