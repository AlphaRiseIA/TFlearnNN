#Ignore if all is squished
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import tflearn
import json
import numpy
import tensorflow
#------------------------------------------------------------
#Open that helpful info for train
with open("info.json") as WorkInfo:
    data = json.load(WorkInfo)
#------------------------------------------------------------
#Now the empty variables that are lonelier than my lost sock
words=[]
tags=[]
auxX=[]
auxY=[]
learning=[]
out=[]
#------------------------------------------------------------
#Irrelevant thing about tokenizing with nltk(my secret father)
for things in data["things"]:
    for rep in things["rep"]:
        auxword = nltk.word_tokenize(rep)
        words.extend(auxword)
        auxX.append(auxword)
        auxY.append(things["tag"])
        if things["tag"] not in tags:
            tags.append(things["tag"])
#------------------------------------------------------------
#Irrelevant and simple thing about my adicction of pineapples on the np array synthesis for send it to the NN and then train it
words = [stemmer.stem(w.lower()) for w in words if w!="?"]
words = sorted(list(set(words)))
tags = sorted(tags)
outNone = [0 for _ in range(len(tags))]
for x, doc in enumerate(auxX):
    pineapple=[]
    auxword= [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in auxword:
            pineapple.append(1)
        else:
            pineapple.append(0)
    fileout = outNone[:]
    fileout[tags.index(auxY[x])]=1
    learning.append(pineapple)
    out.append(fileout)
#Now put all the arrays in simple variables to the NN understand it cause is like a baby and have lerning problems
learning = numpy.array(learning)
out = numpy.array(out)
#------------------------------------------------------------
#The NN here uwu
netPear = tflearn.input_data(shape=[None,len(learning[0])])
netPear = tflearn.fully_connected(netPear,10)
netPear = tflearn.fully_connected(netPear,10)
netPear = tflearn.fully_connected(netPear,len(out[0]),activation="softmax")
netPear = tflearn.regression(netPear)
Pear = tflearn.DNN(netPear)
#------------------------------------------------------------
#Now we cook it and store it in a jar like cookies
coockie .fit(learning,out,n_epoch=1000,batch_size=11,show_metric=True)
coockie .save("CoockiesJar.tflearn")
#------------------------------------------------------------