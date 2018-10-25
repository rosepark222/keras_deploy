import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from tensorflow.python.keras.models import load_model



def init(): 
#	j_file = 'deploy_model_sym53_batch500_epoch1299_1st64_2nd32.json' # deploy_model_epoch599.json'
#	w_file = 'deploy_model_sym53_batch500_epoch1299_1st64_2nd32_weight.h5' # deploy_model_epoch599.h5"
	j_file = 'deploy_balanced.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
	w_file = 'deploy_balanced.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"
#
#	j_file = 'deploy_2_3_x_left_closeB.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
#	w_file = 'deploy_2_3_x_left_closeB.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"
	json_file = open( j_file,'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights( w_file )
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mse', 'accuracy'])
	graph = tf.get_default_graph()
#	j_file20 = 'deploy_x_left_right_c_open_closeB.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
#	w_file20 = 'deploy_x_left_right_c_open_closeB.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"
	j_file20 = 'deploy_15_features_2_3_x_left_closeB.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
	w_file20 = 'deploy_15_features_2_3_x_left_closeB.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"
	json_file20 = open( j_file20,'r')
	loaded_model_json20 = json_file20.read()
	json_file20.close()
	loaded_model20 = model_from_json(loaded_model_json20)
	loaded_model20.load_weights( w_file20 )
	loaded_model20.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mse', 'accuracy'])
	graph20 = tf.get_default_graph()

	j_file53 = 'deploy_15_features_53_sym_hijk.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
	w_file53 = 'deploy_15_features_53_sym_hijk.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"
	json_file53 = open( j_file53,'r')
	loaded_model_json53 = json_file53.read()
	json_file53.close()
	loaded_model53 = model_from_json(loaded_model_json53)
	loaded_model53.load_weights( w_file53 )
	loaded_model53.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mse', 'accuracy'])
	graph53 = tf.get_default_graph()




	#model_json = loaded_model.to_json()
	#with open("./model.json", "w") as json_file:
	#	json_file.write(model_json)
	#loaded_model.save_weights("model.h5")

	return loaded_model,graph, loaded_model20, graph20, loaded_model53, graph53
