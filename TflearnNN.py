import tflearn as tfl
import numpy as np
import tensorflow as tf
#------------------------------------------------------------
#We say the variables for the NN train, you need to fill with the info to process on "x" and on "y"
DATA_1 = np.array(x)
DATA_2 = np.array(y)
#------------------------------------------------------------
#We built up the NN
netPear = tfl.input_data(shape=[1])
netPear = tfl.fully_connected(netPear,3)
netPear = tfl.fully_connected(netPear,3)
netPear = tfl.fully_connected(netPear,1,activation='relu')
netPear = tfl.regression(netPear)
coockie = tfl.DNN(netPear)
#------------------------------------------------------------
#Now we cook it and store it in a jar like cookies
hisotry = coockie.fit(DATA_1,DATA_2,n_epoch=1000,batch_size=11,show_metric=True)
coockie.save("CoockiesJar.tflearn")

