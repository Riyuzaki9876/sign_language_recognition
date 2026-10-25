from flask import Flask, render_template, url_for, request, Response
import sqlite3
#from camera import genarate_frames
import cv2
import os
import shutil
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from keras.models import load_model
import json
import math
from PIL import Image
# import PyAudio
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM admin WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('logged_in.html')

    return render_template('index.html')



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')



@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS admin(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO admin VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('signup.html')
    


@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('index.html')

@app.route('/cam1')
def cam1():

    os.system("python gesture_to_voice.py")
    return render_template('logged_in.html')

@app.route('/cam2')
def cam2():

    os.system("python voice_to_text.py")
    return render_template('logged_in.html')

@app.route('/cam3')
def cam3():

    os.system("python TestEmotionDetector.py")
    return render_template('logged_in.html')

@app.route('/image')
def image():
    # os.system('python virtual_mouse.py')
    return render_template('home.html')


def home():
    return render_template('home.html')

@app.route('/cnn', methods=['GET', 'POST'])
def cnn():
    if request.method == 'POST':
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"

        shutil.copy("indian/test/"+fileName, dst)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'gesturerecognition-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in tqdm(os.listdir(verify_dir)):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 26, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        accuracy=" "
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            #y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'a'
                

            elif np.argmax(model_out) == 1:
                str_label = 'b'
                
            
            elif np.argmax(model_out) == 2:
                str_label = 'c'


            elif np.argmax(model_out) == 3:
                str_label = 'd'


            elif np.argmax(model_out) == 4:
                str_label = 'e'
            
            elif np.argmax(model_out) == 5:
                str_label = 'f'
            
            elif np.argmax(model_out) == 6:
                str_label = 'g'

            elif np.argmax(model_out) == 7:
                str_label = 'h'

            elif np.argmax(model_out) == 8:
                str_label = 'i'

            elif np.argmax(model_out) == 9:
                str_label = 'j'

            elif np.argmax(model_out) == 10:
                str_label = 'k'

            elif np.argmax(model_out) == 11:
                str_label = 'l'

            elif np.argmax(model_out) == 12:
                str_label = 'm'

            elif np.argmax(model_out) == 13:
                str_label = 'n'

            elif np.argmax(model_out) == 14:
                str_label = 'o'

            elif np.argmax(model_out) == 15:
                str_label = 'p'

            elif np.argmax(model_out) == 16:
                str_label = 'q'

            elif np.argmax(model_out) == 17:
                str_label = 'r'

            elif np.argmax(model_out) == 18:
                str_label = 's'

            elif np.argmax(model_out) == 19:
                str_label = 't'

            elif np.argmax(model_out) == 20:
                str_label = 'u'

            elif np.argmax(model_out) == 21:
                str_label = 'v'

            elif np.argmax(model_out) == 22:
                str_label = 'w'

            elif np.argmax(model_out) == 23:
                str_label = 'x'

            elif np.argmax(model_out) == 24:
                str_label = 'y'

            elif np.argmax(model_out) == 25:
                str_label = 'z'

           


                

        return render_template('home.html', cnn_status=str_label,  cnn_ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName)
    return render_template('home.html')


##@app.route('/cam5')
##def cam5():
##
##    os.system("python eye_tracker.py")
##    return render_template('logged_in.html')
##
##@app.route('/cam6')
##def cam6():

    os.system("python audio_part.py")
    return render_template('logged_in.html')

if __name__ == "__main__":
    app.run(debug=True)
