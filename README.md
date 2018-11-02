# Deployment of a Tensorflow Model to the Google cloud

summary: Imagine a kid works on math problems in her iPad and she writes down her answer using Apple Pencil. When iPad recognizes her answer is wrong, it provides relevant mini problems to help her to learn necessary concepts to solve the original problem. Meanwhile, her misconceptions are analyzed from her math equation answer and appropriate feedback is generated for her teachers or parents. This project is a baby step for this lofty goal. An essential technique of this learning platform is handwritten recognition system for math equations. As of today, a bidirectional RNN has been trained to recognize 53 math strokes (not symboles -- details comes below). Additional processing is planned to bring this recognition to symbol and equation level for complete math equation recognition.

I think the best way to have the feel for this project is to dabble with the deployment in Google cloud:      http://pradoxum001.appspot.com/

<!--- ![screenshot from 2018-11-01 08-46-12](https://user-images.githubusercontent.com/38844805/47862556-a495d480-ddb2-11e8-98f0-0a55746f2dbc.png) --> 
This screenshot shows the recognition of digits. Note that 4 and 5 indicate that they are the first part of strokes of 4 and 5.

<img width="412" alt="screen shot 2018-11-02 at 4 04 47 am" src="https://user-images.githubusercontent.com/38844805/47862556-a495d480-ddb2-11e8-98f0-0a55746f2dbc.png">

A simple (way too simple) front end design (predict and clear buttons and a Canvas) was developed using Javascript and HTML, while Python and Flask were used in the back-end framework. The deployment platform for this project is Google cloud  and App engine was very easy to set up. In the front-end, math typesetting is performed any time a recognition is performed and outputs are displayed. The outcome symbols can be either recognized symbols or symbols a stroke belongs to. For example, C can be recognized as the symbol C or right side of two-stroked x. Thus, the recognition output for purportedly C can be either C or X, depending on the features of the stroke (see below). 

<img width="312" height="260" alt="screen shot 2018-11-02 at 4 04 47 am" src="https://user-images.githubusercontent.com/38844805/47912120-bfb92080-de54-11e8-8e7c-ff626ac8abd6.png">

Certain level of misclassificaiton in Machine Learning is inevitable and further processings are required. Math equations are often referred as 2D language and spatial relationships among strokes should be considered for the symbol level or equation level  recognition. Further, the context of an equation in the application (e.g., problem-answer application in a tablet) should provide important infomration to improve the chance of the correct recognition. 

Exciting future plans for the project are:
 1. addition of line detection prior to the training --- I believe line detection is essential because a large set of symbols  are composed up of lines. Intreseting observation is that symbols that are composed up with lines (e.g., + or - or division line) are often separating an entire equation into smaller units. 
 2. develop insights for the spatial relationship among strokes --- math equations are 2D language. Distances and angles matter when we group strokes to form symbols.
 3. develop probabilistic model for relating symbols and strokes --- theories are not always correct but many times useful, thus study techniques discovering the sequence of latent categorical variables (e.g.,  Hidden Markov Model, Viterbi algorithm)
 4. improve stroke recognition --- I need more data for this.  Currently, information for each stroke are heavily unbalanced. I can modify this front-end design and be able to store strokes data from users.


## Credit

I have modified a code from [this](https://youtu.be/f6Bf3gl4hWY) Youtube clip made by Siraj Raval, which was developed to recognize handwritten digit images (MNIST) using [TensorFlow](https://www.tensorflow.org/) and the [Keras](http://keras.io/) Library, wrapped into a Webapp using [Flask](http://flask.pocoo.org/) Micro Framework.

#### Note
There are all three github repositories realted to this project--  
 1) data preparation (https://github.com/rosepark222/HW_Rcode), 
 2) model training (https://github.com/rosepark222/HW_symbol_learn), and 
 3) deployment (https://github.com/rosepark222/keras_deploy). 

