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

	#load in the json list file into a list of dictionaries
	jsonlist = list()
	f = open(trainFile, "r")
	for line in f:
		jsonlist.append(json.loads(line))
	
	#convert inputs to lists
	comment_vectors = list()
	parent_vectors = list()
	labels = list()
	authors = list()
	subreddit = list()
	ups = list()
	downs = list()


	#get information from dicts to make lists
	# print("making lists from dicts from json")
	for d in jsonlist:

		comment_vectors.append( d["comment"] )

		parent_vectors.append( d["parent_comment"] )
		labels.append(d["label"])
		authors.append( d["author"] )
		subreddit.append( d ["subreddit"] )
		ups.append( d["ups"] )
		downs.append( d["downs"] )

	# print("making dataset from lists")

	#setup data to pass into model
	
	train_comments = np.asarray( comment_vectors[: math.floor( len(comment_vectors) * 0.7 ) ] )
	test_comments = np.asarray( comment_vectors[ math.floor( len(comment_vectors) * 0.7 ) + 1 : ] )
	train_labels = np.asarray( labels[ : math.floor( len(labels) * 0.7 ) ] )
	test_labels = np.asarray( labels[ math.floor( len(labels) * 0.7 ) + 1 : ] )

	#make (up, down) tuple and append to list to pass to model input
	updown = list()
	for u, d in zip(ups, downs):
		updown.append( (u, d) )
	train_updown = np.asarray( updown[: math.floor( len(updown) * 0.7 ) ] )
	test_updown = np.asarray( updown[ math.floor( len(updown) * 0.7 ) + 1 : ] )

	# print(np.shape(updown))
	# print (np.shape( train_comments ))
	# print (np.shape(test_comments))
	# print (np.shape(train_labels))
	# print (np.shape(test_labels))


	# code below adapted from tutorial at https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_text_classification.ipynb
	# and https://keras.io/getting-started/sequential-model-guide/#specifying-the-input-shape
	
	print("starting keras")
	
	#model used to create embeddings for sentences

	embeddings = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
	# input layer
	hub_layer = hub.KerasLayer(embeddings, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)


	# setting up keras model
	model = tf.keras.Sequential()
	model.add(hub_layer)
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(4, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.25))

	#model that takes up. down as an input
	model_inout = tf.keras.Sequential()
	model_inout.add(tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)))
	model_inout.add(tf.keras.layers.Dropout(0.25))

	#merge the comment model and the updown model into one layer
	mergedOut = keras.layers.Add()([model.output, model_inout.output])
	#prevent overfitting
	dropout = keras.layers.Dropout(0.2)(mergedOut)
	mergedOut_1 = keras.layers.Dense(8, activation='relu')(dropout)
	mergedOut_2 = keras.layers.Dense(4, activation='relu')(mergedOut_1)
	mergedOut_3 = keras.layers.Dense(1, activation='sigmoid')(mergedOut_2)

	#description of the final model's inputs and outputs
	final_model = keras.models.Model( [model.input, model_inout.input], mergedOut_3 )

	# prints a summary of the layers of the NN
	# final_model.summary()

	final_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



	# train the model
	# low number of epochs for time constraints
	history = final_model.fit([train_comments, train_updown],
                    train_labels,
                    epochs=100,
                    batch_size=51200,
                    validation_data=([test_comments, test_updown], test_labels),
                    verbose=1)

	#this code prints a chart
	
	# history_dict = history.history
	# history_dict.keys()

	# acc = history_dict['accuracy']
	# val_acc = history_dict['val_accuracy']
	# loss = history_dict['loss']
	# val_loss = history_dict['val_loss']

	# epochs = range(1, len(acc) + 1)

	# # "bo" is for "blue dot"
	# plt.plot(epochs, loss, 'bo', label='Training loss')
	# # b is for "solid blue line"
	# plt.plot(epochs, val_loss, 'b', label='Validation loss')
	# plt.title('Training and validation loss')
	# plt.xlabel('Epochs')
	# plt.ylabel('Loss')
	# plt.legend()

	# plt.show()

	# plt.clf()   # clear figure

	# plt.plot(epochs, acc, 'bo', label='Training acc')
	# plt.plot(epochs, val_acc, 'b', label='Validation acc')
	# plt.title('Training and validation accuracy')
	# plt.xlabel('Epochs')
	# plt.ylabel('Accuracy')
	# plt.legend()

	# plt.show()

	model = final_model

def testSarcasm(comment):

	#was used for testing
	# exit()
	# quit()

	#need to pass a nparray to predict method
	# recreate in the same input format as before then pass to predict
	temp_list = list()
	temp_list_2 = list()

	temp_list.append(comment["comment"])
	temp_list_2.append((comment["ups"], comment["downs"] ))
	temp_np = np.asarray(temp_list)
	temp_np_2 = np.asarray(temp_list_2)

	#convert a probability to a boolean
	val = model.predict( [temp_np, temp_np_2] )
	if val[0] > 0.5:
		return True
	else:
		return False