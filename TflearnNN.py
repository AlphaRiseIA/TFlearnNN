#Ignore if all is squished
import nltk
import tflearn as tfl
import numpy as np
import tensorflow as tf
DATA_1 = np.array(x)
DATA_2 = np.array(y)
#------------------------------------------------------------
#The NN here uwu
netPear = tfl.input_data(shape=[None,])
netPear = tfl.fully_connected(netPear,10)
netPear = tfl.fully_connected(netPear,10)
netPear = tfl.fully_connected(netPear,1,activation='relu')
netPear = tfl.regression(netPear)
coockie = tfl.DNN(netPear)
#------------------------------------------------------------
#Now we cook it and store it in a jar like cookies
coockie.fit(DATA_1,DATA_2,n_epoch=1000,batch_size=11,show_metric=True)
coockie.save("CoockiesJar.tflearn")
#------------------------------------------------------------
