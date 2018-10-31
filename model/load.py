import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from tensorflow.python.keras.models import load_model


def compile_model( j_file, w_file ):
	json_file = open( j_file,'r')
	model1_json = json_file.read()
	json_file.close()
	model1 = model_from_json(model1_json)
	model1.load_weights( w_file )
	model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mse', 'accuracy'])
	graph1 = tf.get_default_graph()
	return (model1, graph1)

def init(): 
#	j_file = 'deploy_model_sym53_batch500_epoch1299_1st64_2nd32.json' # deploy_model_epoch599.json'
#	w_file = 'deploy_model_sym53_batch500_epoch1299_1st64_2nd32_weight.h5' # deploy_model_epoch599.h5"
#	j_file = 'deploy_balanced.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
#	w_file = 'deploy_balanced.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"
	j_file = 'deploy_balanced_hand_clean.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
	w_file = 'deploy_balanced_hand_clean.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"
	model1, graph1 = compile_model(j_file, w_file)



#	j_file = 'deploy_2_3_x_left_closeB.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
#	w_file = 'deploy_2_3_x_left_closeB.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"
	# json_file = open( j_file,'r')
	# model1_json = json_file.read()
	# json_file.close()
	# model1 = model_from_json(model1_json)
	# model1.load_weights( w_file )
	# model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mse', 'accuracy'])
	# graph1 = tf.get_default_graph()
#	j_file20 = 'deploy_x_left_right_c_open_closeB.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
#	w_file20 = 'deploy_x_left_right_c_open_closeB.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"
	j_file = 'deploy_model_sym53_batch500_epoch855_1st64_2nd32.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
	w_file = 'deploy_model_sym53_batch500_epoch855_1st64_2nd32_weight.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"

	model2, graph2 = compile_model(j_file, w_file)

	# json_file20 = open( j_file20,'r')
	# model1_json20 = json_file20.read()
	# json_file20.close()
	# model2 = model_from_json(model1_json20)
	# model2.load_weights( w_file20 )
	# model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mse', 'accuracy'])
	# graph2 = tf.get_default_graph()

	#j_file = 'deploy_15_features_53_sym_hijk.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
	#w_file = 'deploy_15_features_53_sym_hijk.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"

	j_file = 'deploy_balanced_hand_clean_2_batch200.json' #deploy_model_sym52_batch500_epoch999_balanced.json' # deploy_model_epoch599.json'
	w_file = 'deploy_balanced_hand_clean_2_batch200.h5' #deploy_model_sym52_batch500_epoch999_balanced_weight.h5' # deploy_model_epoch599.h5"

	model3, graph3 = compile_model(j_file, w_file)
	# json_file53 = open( j_file53,'r')
	# model1_json53 = json_file53.read()
	# json_file53.close()
	# model3 = model_from_json(model1_json53)
	# model3.load_weights( w_file53 )
	# model3.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['mse', 'accuracy'])
	# graph3 = tf.get_default_graph()




	#model_json = model1.to_json()
	#with open("./model.json", "w") as json_file:
	#	json_file.write(model_json)
	#model1.save_weights("model.h5")

	return model1,graph2, model2, graph2, model3, graph3
