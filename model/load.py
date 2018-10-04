import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from tensorflow.python.keras.models import load_model



def init(): 
	json_file = open('deploy_model_epoch599.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("deploy_model_epoch599.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mse', 'accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)


	#loaded_model = load_model('./model_sym61_batch500_epoch1000_1st64_2nd32.h5')
	#loaded_model = load_model('./model_sym61_batch500_epoch1500_1st64_2nd32.h5')
#	loaded_model = load_model('./model_sym20_batch500_epoch600_1st64_2nd32.h5')
	graph = tf.get_default_graph()

	#model_json = loaded_model.to_json()
	#with open("./model.json", "w") as json_file:
	#	json_file.write(model_json)
	#loaded_model.save_weights("model.h5")

	return loaded_model,graph
