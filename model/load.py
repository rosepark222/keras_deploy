import numpy as np
#import keras.models
#from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from tensorflow.python.keras.models import load_model



def init(): 
#	json_file = open('model.json','r')
#	loaded_model_json = json_file.read()
#	json_file.close()
#	loaded_model = model_from_json(loaded_model_json)
#	#load woeights into new model
#	loaded_model.load_weights("model.h5")
#	print("Loaded Model from disk")
#
#	#compile and evaluate loaded model
#	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#	#loss,accuracy = model.evaluate(X_test,y_test)
#	#print('loss:', loss)
#	#print('accuracy:', accuracy)
#

	loaded_model = load_model('./model_sym61_batch500_epoch1000_1st64_2nd32.h5')
	graph = tf.get_default_graph()

	return loaded_model,graph
