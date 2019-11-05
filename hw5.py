"""This is a sample file for hw5. 
It contains the functions that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

import json
import random
import tensorflow as tf

model = None

def trainSarcasm(Model, trainFile):
	global model
	model = Model
	#TODO: process training file here

def testSarcasm(comment):
	return random.choice([True,False])