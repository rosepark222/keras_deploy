# How to Deploy a Keras Model to Production


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
