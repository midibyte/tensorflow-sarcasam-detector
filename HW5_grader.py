"""This is basically the file I will use to grade your hw submissions.
Obviously, the test problems hard-coded in here will be different. You
can use this to test your code. I'm using Python 3.4+.
- Dr. Licato

FILE VERSION: 10/5/19
"""

import random
import traceback
import time
import gensim.models.keyedvectors as word2vec
import json


studentName = "TestStudent"
inputFileName = 'hw5.py'


#load problems
with open("test_nonGraded.jsonlist",'r') as F:
	allComments = [json.loads(l) for l in F.readlines()]

outFile = open("grade_"+studentName+".txt", 'w')

print("Loading w2v pretrained vectors...")
w2vModel = word2vec.KeyedVectors.load_word2vec_format\
 	("GoogleNews-vectors-negative300.bin",binary=True)

def prnt(S):
	global outFile
	outFile.write(str(S) + "\n")
	print(S)

"""This task is graded based on what fraction of the test set you get correct. If you get baselineCorrect, then you get 0. If you get maxCorrect, you get full credit. Extra credit is possible (see code for details).
"""
baselineCorrect = 0.4 #NOTE: These values will not be the same when we determine your grades! We will likely scale based on class performance.
maxCorrect = 0.7
fullCredit = 100
maxScore = 115

#load student file
try:
	F = open(inputFileName, 'r', encoding="utf-8")
	exec("".join(F.readlines()))
except Exception as e:
	prnt("Couldn't open or execute '" + inputFileName + "': " + str(traceback.format_exc()))
	prnt("FINAL SCORE: 0")
	outFile.close()
	exit()

penalty = 0
try:
	prnt("CALLING YOUR trainSarcasm() FUNCTION")
	startTime = time.time()
	trainSarcasm(w2vModel, "train.jsonlist")
	endTime = time.time()
except Exception as e:
	endTime = time.time()
	prnt("\tError arose: " + str(traceback.format_exc()))
	prnt("\tNOTE: We won't penalize you directly for this, but this is likely to lead to exceptions later.")
if endTime - startTime > 5*60:
	prnt("Time to execute was " + str(int(endTime-startTime)) + " seconds; this is too long (-10 points)")
	penalty += 10

numCorrect = 0
for comment in allComments:
	answer = comment['label']
	del comment['label']
	
	prnt("\n\nTESTING ON INPUT PROBLEM:")
	prnt("\t" + json.dumps(comment))
	prnt("CORRECT OUTPUT:")
	print("\t" + str(answer))
	prnt("YOUR OUTPUT:")
	try:
		startTime = time.time()
		result = testSarcasm(comment)
		prnt("\t" + str(result))
		endTime = time.time()		
		if endTime-startTime > 30:
			prnt("Time to execute was " + str(int(endTime-startTime)) + " seconds; this is too long (marked as wrong)")
		elif result==answer:
			prnt("Correct!")
			numCorrect += 1
		else:
			prnt("Incorrect")

	except Exception as e:
		prnt("Marked as incorrect; there was an error while executing this problem: " + str(traceback.format_exc()))
percentCorrect = numCorrect*1.0/len(allComments)
points = min(maxScore, fullCredit * (percentCorrect - baselineCorrect) / (maxCorrect - baselineCorrect))
# print((percentCorrect - baselineCorrect) / (maxCorrect - baselineCorrect))
points = max(0, points)
prnt("\nYou got " + str(percentCorrect*100) + "% correct: +" + str(points) + "/" + str(fullCredit) + " points")
if penalty != 0:
	points -= penalty
	prnt("After penalties: " + str(points))

prnt("=============================")
prnt("=======  FINAL GRADE  =======")
prnt("=============================")
prnt(str(points) + " / 100")

outFile.close()