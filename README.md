# Deployment of a Tensorflow Model to the Google cloud

Overview of the project: Imagine a learner works on math problems in her iPad software and she writes down her answer in a math equation form using Apple Pencil. When education software recognizes an incorrect ansewr, it will provide relevant feedback to her so that she can learn necessary concepts to solve the original problem. Meanwhile, her misconceptions are analyzed from her answer and appropriate insights will be generated for relevant stake holders - her teachers or parents. This project is an initial step for this goal using the latest machine learning technique. An essential technique for this learning ecosystem is Handwritten Recognition System for Math Equations. In this project, a bidirectional RNN has been trained to recognize 53 math strokes (not symbols -- details comes below). Future developments are planned for complete math equation recognition.

The best way to have the feel for this project is to dabble with the deployment in the Google cloud: http://pradoxum001.appspot.com/

---


A simple (may be too simple) front end design (predict and clear buttons and a Canvas) was developed using Javascript and HTML, while Python and Flask were used for the back-end framework. The following screenshot shows the recognition of digits. Note that only first stroke of '4' and '5' were recognized.

<img width="412" alt="screen shot 2018-11-02 at 4 04 47 am" src="https://user-images.githubusercontent.com/38844805/47862556-a495d480-ddb2-11e8-98f0-0a55746f2dbc.png">

Google cloud was chosen as the deployment platform. In the front-end, math typesetting was performed any time a recognition is performed by clicking 'predict' button. The displayed results are either recognized symbols or symbols the stroke may belong to. For example, C can be recognized as the symbol C or x_2_2 of two-stroked x. Thus, a recognition result for a seemingly C stroke can be either C or X.

<img width="312" height="260" alt="screen shot 2018-11-02 at 4 04 47 am" src="https://user-images.githubusercontent.com/38844805/47912120-bfb92080-de54-11e8-8e7c-ff626ac8abd6.png">

Further processings are required for recognizing math equations, which are often considered as 2D language. Thus, spatial (locational) relationships among strokes need to be modelded for the equation level analysis. 

Future plans for the project are:
 1.	Addition of line detection as a preprocess --- I believe the line detection is essential because a large set of symbols are composed up of line(s). Interesting observation is that many symbols that are comprised with lines (e.g., + or - or division line) often divide the entire equation into smaller units, which can be analyzied individually.
 2.	Develop insights for the spatial relationship among strokes --- math equations are 2D language. Distances and angles matter when we group strokes to form symbols.
 3.	Develop probabilistic model for relating symbols and strokes [Muñoz, 2015]--- theories are useful for discovering the sequence of latent categorical variables (e.g., Hidden Markov Model, Viterbi algorithm)
 4.	Improve stroke recognition --- Need for more data. Currently, sample sizes for stroke are heavily unbalanced.


## Credit

I have modified a code from [this](https://youtu.be/f6Bf3gl4hWY) Youtube clip made by Siraj Raval, which was developed to recognize handwritten digit images (MNIST) using [TensorFlow](https://www.tensorflow.org/) and the [Keras](http://keras.io/) Library, wrapped into a Webapp using [Flask](http://flask.pocoo.org/) Micro Framework.

## Reference

Muñoz, F. Á. (2015). Mathematical Expression Recognition based on Probabilistic Grammars (Doctoral dissertation).


#### Note
There are all three github repositories realted to this project--  
 1) data preparation (https://github.com/rosepark222/HW_Rcode), 
 2) model training (https://github.com/rosepark222/HW_symbol_learn), and 
 3) deployment (https://github.com/rosepark222/keras_deploy). 

