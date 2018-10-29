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

#global model, graph, model2, graph2
global model1, graph1, model2, graph2, model3, graph3

import helper as helper
#decoding an image from base64 into raw representation

#TypeError: cannot use a string pattern on a bytes-like object
x61symbol_list= [ "\\sigma_1_1", "(_1_1",       "\\sum_1_1",   "1_1_1",       "n_1_1",       "2_1_1",       ")_1_1",       "r_1_1",
 "i_2_1",       "\\theta_1_1", "\\sum_2_bot", "b_1_1",       "c_1_1",       "4_1_1",       "3_1_1",       "d_1_1",
 "a_1_1",       "8_1_1",       "7_1_1",       "4_2_nose",    "y_1_1",       "0_1_1",       "y_2_flower",  "x_2_left",
 "x_1_1",       "\\sqrt_1_1",  "L_1_1",       "u_1_1",       "\\mu_1_1",    "k_1_1",       "\\lt_1_1",
 "p_1_1",       "p_2_ear",     "q_1_1",       "j_2_1",       "f_2_cobra",   "\\{_1_1",     "\\}_1_1",     "]_1_1",
 "9_1_1",       "h_1_1",       "\\int_1_1",   "t_2_tail",    "e_1_1",       "z_1_1",       "g_1_1",       "s_1_1",
 "5_2_hook",    "6_1_1",       "v_1_1",       "5_1_1",       "w_1_1",       "\\gt_1_1",    "\\alpha_1_1",
 "\\beta_1_1",  "\\gamma_1_1", "m_1_1",       "l_1_1",       "[_1_1",       "\\infty_1_1", "/_1_1"]

xsymbol_list= [
"0_1_1",
"1_1_1",
"2_1_1",
"3_1_1",
"4_1_1",
"5_1_1",
"6_1_1",
"7_1_1",
"8_1_1",
"9_1_1",
"\\sum_1_1",
"\\sqrt_1_1",
"\\int_1_1",
"a_1_1",
"b_1_1",
"c_1_1",
"j_2_1",
"k_1_1",
"n_1_1",
"y_1_1"
 ]

xsymbol_list_53= [ "\\sigma_1_1", "(_1_1",       "\\sum_1_1",   "1_1_1",       "n_1_1",       "2_1_1",       ")_1_1",       "r_1_1",
 "i_2_1",       "\\theta_1_1", "\\sum_2_bot", "b_1_1",       "c_1_1",       "4_1_1",       "3_1_1",       "d_1_1",
 "a_1_1",     "8_1_1",       "7_1_1",       "4_2_nose",    "y_1_1",       "0_1_1",       "x_2_left",
 "x_1_1",       "\\sqrt_1_1",  "o_1_1",       "u_1_1",       "\\mu_1_1",    "k_1_1",      "\\lt_1_1",
  "q_1_1",        "\\{_1_1",     "\\}_1_1",
 "9_1_1",        "\\int_1_1",   "t_2_tail",    "e_1_1",       "x_2_right",       "g_1_1",       "s_1_1",
 "5_2_hook",    "6_1_1",       "v_1_1",       "5_1_1",       "w_1_1",       "\\gt_1_1",    "\\alpha_1_1",
 "\\beta_1_1",  "\\gamma_1_1", "m_1_1",       "l_1_1",        "\\infty_1_1", "/_1_1"]

symbol_list= [ "2_1_1", "3_1_1", "x_2_left", ")_1_1"] 
#symbol_list= [ "x_2_left", "x_2_right", "c_1_1",  ")_1_1", "(_1_1"] 

print(len(xsymbol_list))


#initialize these variables
model1, graph1, model2, graph2, model3, graph3 = init()



def convertImage(imgData1):
	a = 1
	#imgstr = re.search(r'base64,(.*)',imgData1).group(1)
	#print(imgstr)
	#with open('output.png','wb') as output:
		#output.write(imgstr.decode('base64'))
	#	output.write(imgData1)
# def predict_strokes(model, decoded, symbol_list):	
# 	#refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", decoded)
# 	#map_float = np.array( list( map(float, refindout)))
# 	#strokes = np.reshape( map_float , (-1, 2))

# 	strokes = np.reshape( decoded , (-1, 2))

# 	mmm = np.argmin(strokes, axis = 0)
# 	strokes[:,0] =  strokes[:,0] - strokes[ mmm[0] ][0]  
# 	strokes[:,1] =  strokes[:,1] - strokes[ mmm[1] ][1]  	

# 	max_idx = np.argmax(strokes, axis = 0)

# 	scale = max(strokes[ max_idx[0] ][0], strokes[ max_idx[1] ][1])
# 	#print ("max_idx")
# 	#print( max_idx)
# 	strokes[:,0] =  strokes[:,0]/scale * 100
# 	strokes[:,1] =  strokes[:,1]/scale * 100
# 	#print("strokes")
	
# 	x = strokes.reshape(1, len(strokes), 2)
# 	#print(x)
# 	with graph.as_default():
# 		#perform the prediction
# 		out = model.predict(x)
# 		out = out.flatten().tolist()
# 		#print(out)

# 		largest_three_symidx = [out.index(x) for x in sorted(out, reverse=True)[:3]]
# 		print(largest_three_symidx)

# 		largest_three_out = [x for x in sorted(out, reverse=True)[:3]]
# 		print(largest_three_out)

# 		num_1_sym  = symbol_list[ largest_three_symidx[0] ]
# 		num_2_sym  = symbol_list[ largest_three_symidx[1] ]
# 		num_3_sym  = symbol_list[ largest_three_symidx[2] ]
# 		num_1_prob  =  largest_three_out[0]
# 		num_2_prob  =  largest_three_out[1]
# 		num_3_prob  =  largest_three_out[2]

# 		#print(largest_three)
# 		arg_max = np.argmax(out)
# 		#print( arg_max )
# 		predicted = symbol_list[ arg_max ]

# 		#print ("pred: symbol[%d]=         %s  ; len = %d" %(arg_max, predicted, len(strokes)))
# 		predicted = " [%s, %s, %s]= [%3.2f,%3.2f,%3.2f] = %d" %(num_1_sym, num_2_sym, num_3_sym,
# 			num_1_prob, num_2_prob, num_3_prob , len(strokes))
# 		print (predicted)
# 		#symbol_list[]
# 		#convert the response to a string
# 		#response = np.array_str(np.argmax(out,axis=1))
# 		return predicted #response	


# def decode_strokes(model, decoded, symbol_list):
# 	a = 1
# 	#print("request.get_data()")
# 	print(decoded)
# 	print(type(decoded))	#print("100")
# 	refindout = re.findall(r"[-+]?[0-9]*\.?[0-9]+", decoded)
# 	decoded = np.array( list( map(float, refindout)))
# 	print(decoded)
# 	print(type(decoded))	#print("100")
# 	#print(type("100"))
# 	#print(imgData2.decode("UTF-8"))
# 	print("-------------------")
# 	print(decoded)
# 	npa = np.array(decoded)
# 	print("-------------------")
# 	print(npa)
# 	print(type(npa))
# 	spl = np.flatnonzero( npa == -999)
# 	print(spl)
# 	strks = np.split(npa, spl)[1:] #split and throw away the empty first
# 	print(strks)
# 	strokes = [x[1:] for x in strks] #remove -999 from strokes
# 	out_string = ''
# 	for s in strokes:
# 		out_string = out_string + predict_strokes(model, s, symbol_list) + "================>"
# #================>
# #++++++++++++++++> 
# 	return (out_string)

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

	
@app.route('/predict1/',methods=['GET','POST'])
def predict2():
	imgData2 = request.get_data() 
	#decoded = imgData2.decode("UTF-8")
	with graph1.as_default():
		out_str = helper.deploy_predict_online_stroke_data (imgData2, model1, xsymbol_list_53)
	return (out_str) 

	#print( "%d was sent to python"% (len(decoded)))

	#return (decoded + "1001")
@app.route('/predict2/',methods=['GET','POST'])
def predict15():
	print("predict2")
	imgData2 = request.get_data() 
	#decoded = imgData2.decode("UTF-8")
	with graph2.as_default():
		out_str = helper.deploy_predict_online_stroke_data (imgData2, model2, xsymbol_list_53)
	return (out_str) 

@app.route('/predict3/',methods=['GET','POST'])
def predict53():
	print("predict3")
	imgData2 = request.get_data() 
	#decoded = imgData2.decode("UTF-8")
	#with graph3.as_default():
	#	out_str = helper.deploy_predict_online_stroke_data (imgData2, model3, xsymbol_list_53)
	out_str = " $$x = { -b + \\sqrt{b^2-4ac} \\over 2a}.$$ "
	return (out_str) 


if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)

# @app.route('/predict/',methods=['GET','POST'])
# def predict():
# 	#whenever the predict method is called, we're going
# 	#to input the user drawn character as an image into the model
# 	#perform inference, and return the classification
# 	#get the raw data format of the image
# 	imgData = request.get_data() #UTF-16 data
# 	print ("type(imgData)")   #"debug")
# 	print (type(imgData))   #"debug")
# 	#imgData = imgData.decode('utf-8')
# 	#encode it into a suitable format
# 	convertImage(imgData)
# 	print ("debug")
# 	#read the image into memory
# 	x = imread('output.png',mode='L')
# 	#compute a bit-wise inversion so black becomes white and vice versa
# 	x = np.invert(x)
# 	#make it the right size
# 	x = imresize(x,(28,28))
# 	#imshow(x)
# 	#convert to a 4D tensor to feed into our model
# 	x = x.reshape(1,28,28,1)
# 	print ("debug2")
# 	#in our computation graph
# 	with graph.as_default():
# 		#perform the prediction
# 		out = model.predict(x)
# 		print(out)
# 		print(np.argmax(out,axis=1))
# 		print ("debug3")
# 		#convert the response to a string
# 		response = np.array_str(np.argmax(out,axis=1))
# 		return response	

# m_1_1
# gamma_1_1
# beta_1_1
# alpha_1_1
# 5_2_hook
# w_1_1

# s_1_1
#  [g_1_1, \sum_1_1, )_1_1]= [0.38,0.24,0.12] = 56
# 127.0.0.1 - - [30/Sep/2018 10:32:05] "POST /predict2/ HTTP/1.1" 200 -
#  [g_1_1, 5_2_hook, 9_1_1]= [0.82,0.04,0.04] = 58
# 127.0.0.1 - - [30/Sep/2018 10:32:08] "POST /predict2/ HTTP/1.1" 200 -
#  [g_1_1, \sum_1_1, 5_2_hook]= [0.75,0.08,0.04] = 46

# e_1_1
#  [0_1_1, \gamma_1_1, \theta_1_1]= [0.26,0.12,0.09] = 56
# 127.0.0.1 - - [30/Sep/2018 10:30:40] "POST /predict2/ HTTP/1.1" 200 -
#  [0_1_1, \theta_1_1, a_1_1]= [0.35,0.21,0.10] = 59
# 127.0.0.1 - - [30/Sep/2018 10:30:44] "POST /predict2/ HTTP/1.1" 200 -
#  [\theta_1_1, 0_1_1, 6_1_1]= [0.50,0.16,0.09] = 57
# 127.0.0.1 - - [30/Sep/2018 10:30:47] "POST /predict2/ HTTP/1.1" 200 -
#  [0_1_1, \theta_1_1, a_1_1]= [0.53,0.15,0.06] = 50
# 127.0.0.1 - - [30/Sep/2018 10:30:51] "POST /predict2/ HTTP/1.1" 200 -
#  [i_2_1, \sum_1_1, \sum_2_bot]= [0.26,0.19,0.11] = 49
# 127.0.0.1 - - [30/Sep/2018 10:30:54] "POST /predict2/ HTTP/1.1" 200 -
#  [\sum_1_1, \sum_2_bot, b_1_1]= [0.45,0.15,0.07] = 56
# 127.0.0.1 - - [30/Sep/2018 10:30:57] "POST /predict2/ HTTP/1.1" 200 -
#  [\theta_1_1, 0_1_1, \gamma_1_1]= [0.51,0.22,0.07] = 57



# t_2_tail

# h_1_1
#  [k_1_1, h_1_1, \sum_2_bot]= [0.23,0.22,0.21] = 60
#  [k_1_1, h_1_1, b_1_1]= [0.60,0.34,0.03] = 56
#  [k_1_1, h_1_1, b_1_1]= [0.78,0.17,0.02] = 49
#  [a_1_1, 6_1_1, 5_1_1]= [0.30,0.16,0.13] = 58
#  [k_1_1, 6_1_1, h_1_1]= [0.45,0.14,0.12] = 53
#  [\sum_2_bot, k_1_1, h_1_1]= [0.30,0.21,0.20] = 48
#  [\sum_2_bot, 6_1_1, c_1_1]= [0.47,0.20,0.06] = 58

# int  f_2_cobra

#  [g_1_1, \int_1_1, y_1_1]= [0.50,0.21,0.10] = 34
#  [g_1_1, \int_1_1, )_1_1]= [0.63,0.10,0.10] = 40
#  [y_1_1, \int_1_1, g_1_1]= [0.43,0.26,0.19] = 33
#  [y_1_1, \int_1_1, g_1_1]= [0.42,0.36,0.11] = 39

# j_2_1

#  [)_1_1, j_2_1, \int_1_1]= [0.52,0.21,0.12] = 30
#  [y_1_1, z_1_1, \}_1_1]= [0.50,0.12,0.08] = 44
#  [)_1_1, j_2_1, \int_1_1]= [0.68,0.11,0.09] = 45
#  [\int_1_1, (_1_1, )_1_1]= [0.26,0.22,0.14] = 48
#  [)_1_1, ]_1_1, j_2_1]= [0.44,0.22,0.15] = 53
#  [y_1_1, z_1_1, (_1_1]= [0.50,0.12,0.12] = 51


# q_1_1
#  [a_1_1, g_1_1, \sum_1_1]= [0.72,0.23,0.02] = 70
#  [g_1_1, \sum_1_1, a_1_1]= [0.39,0.14,0.14] = 74
#  [g_1_1, q_1_1, 5_1_1]= [0.90,0.05,0.01] = 48
#  [g_1_1, \sum_1_1, q_1_1]= [0.85,0.05,0.02] = 59


# p_2_ear

# p_1_1  

#  [n_1_1, 2_1_1, \sqrt_1_1]= [1.00,0.00,0.00] = 69 ---- > left bar down first drawn
#  [\sqrt_1_1, r_1_1, f_2_cobra]= [0.96,0.02,0.01] = 53
#  [\sqrt_1_1, n_1_1, r_1_1]= [0.99,0.01,0.00] = 73

#  [p_1_1, \mu_1_1, 3_1_1]= [0.48,0.14,0.07] = 68    ----> right ball first drawn
#  [\mu_1_1, p_1_1, u_1_1]= [0.36,0.29,0.14] = 71
#  [\beta_1_1, 0_1_1, p_1_1]= [0.46,0.33,0.10] = 68
#  [p_1_1, \theta_1_1, \beta_1_1]= [0.65,0.14,0.06] = 61
#  [p_1_1, \theta_1_1, \sigma_1_1]= [0.77,0.13,0.03] = 53

# \\gt_1_1   === p pops up when written fast , works well when written slowly
# \\lt_1_1   === I used right hand of K because of small SS. but I cannot make it work 
# 				 remove it
#                [\sum_1_1, e_1_1, L_1_1]= [0.97,0.01,0.01]



# k_1_1 ==== 

 # [h_1_1, k_1_1, a_1_1]= [0.41,0.33,0.16] = 106
 # [k_1_1, h_1_1, b_1_1]= [0.82,0.08,0.07] = 81
 # [a_1_1, h_1_1, k_1_1]= [0.59,0.18,0.10] = 76
 # [b_1_1, \sum_1_1, 5_2_hook]= [0.73,0.09,0.06] = 84
 # [k_1_1, b_1_1, \sum_1_1]= [0.37,0.20,0.18] = 87
 # [k_1_1, h_1_1, b_1_1]= [0.82,0.09,0.08] = 113

# \\mu_1_1  === not working

# u_1_1  === u with right tail -> a_1_1
# 			u without right tail -> i_2_1

# L_1_1  === cannot draw this! 

# x_1_1  === cannot draw this! 
#  [n_1_1, 7_1_1, 1_1_1]= [0.78,0.08,0.03] = 67
#  [n_1_1, 7_1_1, 1_1_1]= [0.97,0.01,0.00] = 85
#  [n_1_1, p_1_1, \sqrt_1_1]= [0.97,0.02,0.01] = 36
#  [n_1_1, 1_1_1, 7_1_1]= [0.65,0.13,0.09] = 71
#  [n_1_1, 7_1_1, 1_1_1]= [0.36,0.32,0.25] = 76
#  [n_1_1, 7_1_1, 1_1_1]= [0.79,0.09,0.05] = 60



# m_1_1  ==== recongized as nn is the preceding pattern of m
# \\beta_1_1 ==== not enough stroke data for beta... remove it!!!!
# 
#cannot make it work at all:

# t_2_tail ==== could draw this at all. predicts others unrelated symbols [k_1_1, b_1_1, \sum_2_bot]= [0.38,0.33,0.07]
#              feels like it provides little information and confused RNN
#"\\sigma_1_1",  [\infty_1_1, 8_1_1, \sqrt_1_1]= [0.72,0.11,0.08]


# \\{_1_1     ==== recognized as \\sum_1_1    ---- this makes sense because regulariztion makes { looks like \\sum
# \\}_1_1     ==== recognized as 3_1_1    ---- this makes sense because regularization makes } looks like 3
# [_1_1       ==== recognized as \\sum_1_1    ---- this makes sense 
# ]_1_1       ==== correctly recognized but not useful unless [_1_1 is recognized as well
# (_1_1       ==== recognized as \\sum_1_1    ---- this makes sense 
# )_1_1       ==== recongized as \\gt_1_1 
# 1_1_1       ==== sometimes work with short nose but most of times recognized as n with long nose
# 4_1_1       ==== not working
# 5_2_hook    ==== {h, b, n, t_2_tail, etc}
# 5_1_1       ==== {g_1_1, 5_1_1}  if I remove g_1_1, then would RNN undersntand it as 5_1_1 (I think so)


#6_1_1   ====  {6_1_1, b_1_1} ... confused between thses two.
#This is somewhat serious problem. 

#7_1_1   ====  something wrong..... not working at all.....
#9_1_1   ====  [g_1_1, y_1_1, 9_1_1]= [0.50,0.26,0.19]
#r_1_1   ====  recognized as sqrt   [\sqrt_1_1, r_1_1, y_1_1]= [0.93,0.06,0.01]   
#"i_2_1"
# g_1_1 ==== works fine but this overshadow 9

# #theta  =====> 
#  [a_1_1, g_1_1, \theta_1_1]= [0.82,0.07,0.04]
#  [a_1_1, g_1_1, \theta_1_1]= [0.47,0.12,0.11]
#  [6_1_1, a_1_1, g_1_1]= [0.23,0.23,0.21]
#  [6_1_1, \theta_1_1, b_1_1]= [0.34,0.21,0.19]
#  [a_1_1, \theta_1_1, g_1_1]= [0.39,0.13,0.11]
#  [6_1_1, a_1_1, \theta_1_1]= [0.23,0.23,0.20]
#  [a_1_1, \theta_1_1, 6_1_1]= [0.25,0.19,0.15]
#  [a_1_1, b_1_1, 6_1_1]= [0.49,0.16,0.10]
#  [b_1_1, 6_1_1, a_1_1]= [0.24,0.21,0.17]
#  [\theta_1_1, 0_1_1, q_1_1]= [0.49,0.18,0.09]
#  [0_1_1, \theta_1_1, g_1_1]= [0.23,0.23,0.21]


# # C =====> part of 6, 0, sum 

#  [\sum_1_1, k_1_1, \sum_2_bot]= [0.36,0.16,0.13]
#  [6_1_1, \sum_1_1, b_1_1]= [0.35,0.22,0.14]
#  [0_1_1, 1_1_1, a_1_1]= [0.59,0.08,0.04]
#  [6_1_1, b_1_1, \theta_1_1]= [0.38,0.27,0.09]
#  [6_1_1, b_1_1, \sum_1_1]= [0.67,0.16,0.07]
#  [\sum_1_1, 6_1_1, b_1_1]= [0.45,0.16,0.14]
#  [b_1_1, k_1_1, 6_1_1]= [0.58,0.14,0.13]
#  [6_1_1, b_1_1, \sum_1_1]= [0.58,0.29,0.03]
#  [6_1_1, \sum_2_bot, b_1_1]= [0.25,0.19,0.19]
#  [0_1_1, 5_2_hook, \beta_1_1]= [0.23,0.10,0.09]



# x_2_left ====> n begins with x_2_left , RNN determines mind early that the symbol is n by looking at the beginning half of the the stroke?
# almost 100% recognize as n =====> 
# A similar pattern occurs at the 6, first half of the 6 is C and C -> 6; both 800 samples. 
# A similar pattern occurs when I draw a flipped 3 (epsilon), it recognize it as \\sum, because first half essentially drawing C 
# Simple patterns often part of more complex pattern and RNN recognized it as the part of more complex featured symbol ?  
# Same token, (_1_1  has virtually no feature, it can be part of 6, sum, gamm, which contains that strokes. ----> limitation of RNN or effect of normalization that removed important feature?

#  [n_1_1, p_1_1, r_1_1]= [1.00,0.00,0.00]
#  [n_1_1, p_1_1, \gt_1_1]= [0.92,0.03,0.02]
#  [n_1_1, m_1_1, y_1_1]= [0.99,0.00,0.00]
#  [n_1_1, p_1_1, r_1_1]= [0.99,0.01,0.00]
#  [n_1_1, 7_1_1, 1_1_1]= [0.87,0.06,0.05]
#  [n_1_1, m_1_1, y_1_1]= [0.99,0.00,0.00]
#  [n_1_1, m_1_1, 1_1_1]= [0.87,0.12,0.00]
#  [n_1_1, p_1_1, m_1_1]= [0.99,0.00,0.00]
#  [n_1_1, p_1_1, r_1_1]= [0.99,0.01,0.00]
 


# 4_2_nose  ==== little feature
# y_2_flower


# y_1_1  === seems working fine
# h_1_1
# g_1_1  == 9 5
# \\}_1_1
# \\{_1_1
# [_1_1
# ]_1_1



