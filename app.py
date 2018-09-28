#our web app framework!
import csv
  
#   https://keras.io/getting-started/sequential-model-guide/#examples
#  check this out
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs
#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs#  check this out for shaping input and outputs

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import load_model

 
#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import * 
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

#decoding an image from base64 into raw representation

#TypeError: cannot use a string pattern on a bytes-like object
symbol_list= [ "\\sigma_1_1", "(_1_1",       "\\sum_1_1",   "1_1_1",       "n_1_1",       "2_1_1",       ")_1_1",       "r_1_1",
 "i_2_1",       "\\theta_1_1", "\\sum_2_bot", "b_1_1",       "c_1_1",       "4_1_1",       "3_1_1",       "d_1_1",
 "a_1_1",       "8_1_1",       "7_1_1",       "4_2_nose",    "y_1_1",       "0_1_1",       "y_2_flower",  "x_2_left",
 "x_1_1",       "\\sqrt_1_1",  "L_1_1",       "u_1_1",       "\\mu_1_1",    "k_1_1",       "\\lt_1_1",
 "p_1_1",       "p_2_ear",     "q_1_1",       "j_2_1",       "f_2_cobra",   "\\{_1_1",     "\\}_1_1",     "]_1_1",
 "9_1_1",       "h_1_1",       "\\int_1_1",   "t_2_tail",    "e_1_1",       "z_1_1",       "g_1_1",       "s_1_1",
 "5_2_hook",    "6_1_1",       "v_1_1",       "5_1_1",       "w_1_1",       "\\gt_1_1",    "\\alpha_1_1",
 "\\beta_1_1",  "\\gamma_1_1", "m_1_1",       "l_1_1",       "[_1_1",       "\\infty_1_1", "/_1_1"]


def convertImage(imgData1):
	#imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	with open('output.png','wb') as output:
		#output.write(imgstr.decode('base64'))
		output.write(imgData1)
	

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data() #UTF-16 data
	print ("type(imgData)")   #"debug")
	print (type(imgData))   #"debug")
	#imgData = imgData.decode('utf-8')
	#encode it into a suitable format
	convertImage(imgData)
	print ("debug")
	#read the image into memory
	x = imread('output.png',mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	print ("debug2")
	#in our computation graph
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		print ("debug3")
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	
@app.route('/predict2/',methods=['GET','POST'])
def predict2():
	imgData2 = request.get_data() 
	#print("request.get_data()")
	#print(imgData2)
	#print(type(imgData2))
	#print("100")
	#print(type("100"))
	#print(imgData2.decode("UTF-8"))
	decoded = imgData2.decode("UTF-8")
	#print( "%d was sent to python"% (len(decoded)))
	refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", decoded)
	map_float = np.array( list( map(float, refindout)))
	strokes = np.reshape( map_float , (-1, 2))

	mmm = np.argmin(strokes, axis = 0)
	strokes[:,0] =  strokes[:,0] - strokes[ mmm[0] ][0]  
	strokes[:,1] =  strokes[:,1] - strokes[ mmm[1] ][1]  	

	max_idx = np.argmax(strokes, axis = 0)
	#print ("max_idx")
	#print( max_idx)
	strokes[:,0] =  strokes[:,0]/strokes[ max_idx[0] ][0] * 100
	strokes[:,1] =  strokes[:,1]/strokes[ max_idx[1] ][1] * 100
	#print("strokes")
	#print(strokes)
	x = strokes.reshape(1, len(strokes), 2)
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		#print(out)
		arg_max = np.argmax(out)
		#print( arg_max )
		predicted = symbol_list[ arg_max ]
		print ("pred: symbol[%d]=         %s  ; len = %d" %(arg_max, predicted, len(strokes)))
		#symbol_list[]
		#convert the response to a string
		#response = np.array_str(np.argmax(out,axis=1))
		return predicted #response	

	#return (decoded + "1001")

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
