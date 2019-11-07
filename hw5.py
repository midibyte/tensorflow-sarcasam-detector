"""This is a sample file for hw5. 
It contains the functions that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

#pip3 install tensorflow
#pip3 install tensorflow-hub
#pip3 install tensorflow-datasets
#pip3 install matplotlib

import json
import random
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds 
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

model = None

"""
– “label” – A Boolean value saying whether this is a sarcastic comment
– “comment” – the text of the comment itself
– “author” – the username of the comment’s author
– “subreddit” – the name of the subreddit this comment came from
– “ups” / “downs” – the # of upvotes / downvotes this comment got
– ”parent_comment” – the text of the comment this one replied to
"""

embeddings = None

def trainSarcasm(Model, trainFile):
	global model
	global embeddings
	# model = Model
	#TODO: process training file here
	#load in the json list file into a list of dictionaries
	jsonlist = list()
	f = open(trainFile, "r")
	for line in f:
		jsonlist.append(json.loads(line))
	
	#convert words to vectors
	comment_vectors = list()
	parent_vectors = list()
	labels = list()
	authors = list()
	subreddit = list()
	ups = list()
	downs = list()

	# #get information from dicts 
	# for d in jsonlist:
	# 	w_vec_list = list()
	# 	for w in d["comment"]:
	# 		vector = model[w]
	# 		w_vec_list.append(vector)
	# 	#comment vector - mean of word vectors
	# 	s_vec = sum(w_vec_list) / len(w_vec_list)
	# 	comment_vectors.append(s_vec)
		
	# 	#parent comment vector
	# 	parent_w_vec_list = list()
	# 	for w in d["parent_comment"]:
	# 		vector = model[w]
	# 		parent_w_vec_list.append(vector)
	# 	parent_vec = sum(parent_w_vec_list) / len(parent_w_vec_list)
	# 	parent_vectors.append(parent_vec)
	# 	labels.append(d["label"])
	# 	authors.append(model[d["author"]])
	# 	subreddit.append( model[ d ["subreddit"] ] )
	# 	ups.append( d["ups"] )
	# 	downs.append( d["downs"] )

	#get information from dicts 
	print("making lists from dicts from json")
	for d in jsonlist:

		comment_vectors.append( d["comment"] )

		parent_vectors.append( d["parent_comment"] )
		labels.append(d["label"])
		authors.append( d["author"] )
		subreddit.append( d ["subreddit"] )
		ups.append( d["ups"] )
		downs.append( d["downs"] )
	# print("making 2d vector")
	#need to make a 2d vector to pass to keras
	# input_vec = np.column_stack( (comment_vectors, parent_vectors, labels, authors, subreddit, ups, downs) )

	# print(np.shape(input_vec))
	print("making dataset from lists")
	#make a dataset to pass to keras
	
	train_comments = np.asarray( comment_vectors[: math.floor( len(comment_vectors) * 0.7 ) ] )
	test_comments = np.asarray( comment_vectors[ math.floor( len(comment_vectors) * 0.7 ) + 1 : ] )
	train_labels = np.asarray( labels[ : math.floor( len(labels) * 0.7 ) ] )
	test_labels = np.asarray( labels[ math.floor( len(labels) * 0.7 ) + 1 : ] )

	train_subreddit = np.asarray( subreddit[: math.floor( len(comment_vectors) * 0.7 ) ] )
	test_subreddit = np.asarray( subreddit[ math.floor( len(comment_vectors) * 0.7 ) + 1 : ] )

	updown = list()
	for u, d in zip(ups, downs):
		updown.append( (u, d) )
	# updown = np.asarray(updown)
	train_updown = np.asarray( updown[: math.floor( len(updown) * 0.7 ) ] )
	test_updown = np.asarray( updown[ math.floor( len(updown) * 0.7 ) + 1 : ] )

	print(np.shape(updown))
	print (np.shape( train_comments ))
	print (np.shape(test_comments))
	print (np.shape(train_labels))
	print (np.shape(test_labels))


	# code adapted from tutorial at https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb
	
	print("starting keras")
	
	#model used to create embeddings for sentences

	embeddings = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
	# input layer
	hub_layer = hub.KerasLayer(embeddings, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)
	hub_layer_1 = hub.KerasLayer(embeddings, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)

	# hub_layer(train_comments[:3])

	# setting up keras model
	model = tf.keras.Sequential()
	model.add(hub_layer)
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(4, activation='relu'))
	# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

	# #train on subreddits here
	# model_s = tf.keras.Sequential()
	# model_s.add(hub_layer_1)
	# model_s.add(tf.keras.layers.Dense(16, activation='relu'))
	# # model_s.add(tf.keras.layers.Dense(1, activation='sigmoid'))

	model_inout = tf.keras.Sequential()
	model_inout.add(tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)))

	mergedOut = keras.layers.Add()([model.output, model_inout.output])
	mergedOut_1 = keras.layers.Dense(8, activation='relu')(mergedOut)
	mergedOut_2 = keras.layers.Dense(1, activation='sigmoid')(mergedOut_1)

	final_model = keras.models.Model( [model.input, model_inout.input], mergedOut_2 )

	# prints a summary of the layers of the NN
	final_model.summary()

	final_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



	# train the model
	# low number of epochs for testing
	history = final_model.fit([train_comments, train_updown],
                    train_labels,
                    epochs=6,
                    batch_size=512,
                    validation_data=([test_comments, test_updown], test_labels),
                    verbose=1)
	
	history_dict = history.history
	history_dict.keys()

	acc = history_dict['accuracy']
	val_acc = history_dict['val_accuracy']
	loss = history_dict['loss']
	val_loss = history_dict['val_loss']

	epochs = range(1, len(acc) + 1)

	# "bo" is for "blue dot"
	plt.plot(epochs, loss, 'bo', label='Training loss')
	# b is for "solid blue line"
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()

	plt.show()

	plt.clf()   # clear figure

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()

	plt.show()

	model = final_model

def testSarcasm(comment):

	# exit()
	# quit()

	#need to pass a nparray to predict
	temp_list = list()
	temp_list_2 = list()

	temp_list.append(comment["comment"])
	temp_list_2.append((comment["ups"], comment["downs"] ))
	temp_np = np.asarray(temp_list)
	temp_np_2 = np.asarray(temp_list_2)



	val = model.predict( [temp_np, temp_np_2] )
	if val[0] > 0.5:
		return True
	else:
		return False