# How to Deploy a Keras Model to Production
\\deployment to google cloud
title: online handwritten mathematics equation recognition (either browser based or tablet based education products).
summary: Imagine a kid works on math problems in her iPad and she writes down her answer using Apple Pencil. When iPad recognizes her answer is wrong, it provides relevant mini problems to help her to learn necessary concepts to solve the original problem. Meanwhile, her misconceptions are analyzed from her math equation answer and appropriate feedback is generated for her teachers or parents. This project is a baby step for this lofty goal. An essential technique of this learning platform is handwritten recognition system for math equations. As of today, a bidirectional RNN has been trained to recognize 53 math strokes (not symboles -- details comes below). Additional processing is planned to bring this recognition to symbol and equation level for complete math equation recognition.
There are all three github repositories --  1) data preparation (https://github.com/rosepark222/HW_Rcode), 2) model training (https://github.com/rosepark222/HW_symbol_learn), and 3) deployment (https://github.com/rosepark222/keras_deploy). 
I think the best way to have the feel for this project is to dabble with the deployment in Google cloud: http://pradoxum001.appspot.com/

details: 1) developed Flask back-end using Tensorflow, 2) developed web client in javascript, 3) deployed the front and back-end to Google cloud 


A simple (way too simple) front end design was developed using Javascript and HTML, while Python and Flask were used in the back-end framework. The deployment platform for this project is Google cloud since it is free for one year and App engine was easy to set up. In the web page, math typesetting is performed any time a recognition occurs. The outcome symbols can be either recognized symbols or most likely symbol the stroke is part of. For example, C can be recognized as the symbol C or right side of two stroke x. Thus, the recognition output for purportedly C 
can be either C or X or even open bracket (, depending on the features of the stroke. This is the reason context of the equation is necessary to correctly recognize the entire equation. Certain level of ambiguity under ML is inevitable and spatial relationship among strokes should be incorporated into the symbol or equation recognition. I am thrilled to continue investigating theoretical and practical solutions in this challenging project. 


Exciting future directions for the project are:
1. addition of line detection prior to the training --- I believe line detections are easy yet essential in the recognition of a large set of symbols that are composed up of lines. 
2. develop insights for the spatial relationship among strokes --- math equations are 2D language. Distances and angles matter when we group strokes to form symbols.
3. develop probabilistic model for relating symbols and strokes --- theories are not always correct but many times useful, thus study more on Hidden Markov Model, Viterbi algorithm.
4. improve stroke recognition --- I need more data, data, data!  Currently, information for each stroke are either lacking or heavily unbalanced even though data are 'artificially' balanced by duplications.



## Overview

This is the code for [this](https://youtu.be/f6Bf3gl4hWY) video on Youtube by Siraj Raval. We're going to build a model that recognizes handwritten digit images (MNIST).  Developed using [TensorFlow](https://www.tensorflow.org/) and the super simple [Keras](http://keras.io/) Library. Wrapped into a Webapp using [Flask](http://flask.pocoo.org/) Micro Framework.

## Dependencies

```sudo pip install -r requirements.txt```

## Usage

Once dependencies are installed, just run this to see it in your browser. 

```python app.py```

That's it! It's serving a saved Keras model to you via Flask. 

## Credits

The credits for this code go to [moinudeen](https://github.com/moinudeen). I've merely created a wrapper to get people started.


 
gcloud app deploy  main.yaml 

![screenshot from 2018-11-01 08-46-12](https://user-images.githubusercontent.com/38844805/47862556-a495d480-ddb2-11e8-98f0-0a55746f2dbc.png)
