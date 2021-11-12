import pickle
import math
from flask import Flask, request, render_template
import numpy as np
import os
from tensorflow import keras
import cv2
from werkzeug.utils import secure_filename

upload_folder = 'static/uploaded'
model = keras.models.load_model('resnet50_model.h5')

app = Flask(__name__, template_folder='.')

app.config['upload_folder'] = upload_folder

def preprocess_imgs_test(img, img_size):
    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
    return np.array(img)

@app.route('/')
@app.route('/home')
def index():
	return render_template("index.html")

@app.route('/login')
def login():
	return render_template('login.html')

@app.route('/heart_dieseas')
def heart_dis():
	return render_template('heart_dieseas.html')

@app.route('/brain')
def brain_det():
	return render_template('brain_tumor.html')

@app.route('/kidney_dieseas')
def kidney_dis():
	return render_template('kidney_dieseas.html')

@app.route('/diabetes')
def diabetes():
	return render_template('diabetes.html')

@app.route('/heart_dieseas_predict', methods=['POST', 'GET'])
def predict_heart_diesease():
	heart_model = pickle.load(open('heart_KNN_model.pkl', 'rb'))
	heart_scale_values = pickle.load(open('heart_scaler.pkl', 'rb'))

	features = [float(x) for x in request.form.values()]
	categorical_val = features[8:]
	del features[8:]
	#for cp
	for i in range(1,5):
		if int(categorical_val[0]) == i:
			features.append(1)
		else:
			features.append(0)
	#for restecg
	for i in range(3):
		if int(categorical_val[1]) == i:
			features.append(1)
		else:
			features.append(0)
	#for slope
	for i in range(1,4):
		if int(categorical_val[2]) == i:
			features.append(1)
		else:
			features.append(0)
	#for ca
	for i in range(5):
		if int(categorical_val[3]) == i:
			features.append(1)
		else:
			features.append(0)
	#for thal
	for i in range(4):
		if int(categorical_val[4]) == i:
			features.append(1)
		else:
			features.append(0)
	cols_index = [0,2,3,5,7] #came from colab model file look into that for more information

	temp = []
	for i in cols_index:
	  temp.append(features[i])
	temp = heart_scale_values.transform(np.array(temp).reshape(1,-1))
	temp = temp.ravel()
	j = 0
	for i in cols_index:
	  features[i] = temp[j]
	  j += 1 
	result = heart_model.predict([features])[0] 
	if result == 0:
		return render_template("heart_dieseas.html", result='Cheers, Patient have a good Heart !!!')
	else:
		return render_template("heart_dieseas.html", result='Patient have Heart Diesease, Do not panic')

@app.route('/brain_tumor_predict', methods=['POST', 'GET'])
def b_t_prediction():
	img = request.files['file']
	img.save(os.path.join(app.config['upload_folder'], img.filename))
	img_prep = preprocess_imgs_test(cv2.imread('static/uploaded/'+img.filename), (224,224))
	result = model.predict(np.expand_dims(img_prep, axis=0))
	print(result)
	if result[0][0] < 0.5:
		return render_template('brain_tumor.html', result='Patient not has Brain Tumor')
	else:
		return render_template('brain_tumor.html', result='Patient has Brain Tumor')

@app.route('/kidney_dieseas_predict', methods=['POST', 'GET'])
def predict_kidney_diesease():
	kidney_model = pickle.load(open('kidney_model.pkl', 'rb'))
	kidney_scale_values = pickle.load(open('kidney_scaler.pkl', 'rb'))
	
	features = [float(x) for x in request.form.values()]
	cols_index = [0, 2, 6, 4, 9, 3] #came from colab model file look into that for more information

	temp = []
	for i in cols_index:
	  temp.append(features[i])
	temp = kidney_scale_values.transform(np.array(temp).reshape(1,-1))
	temp = temp.ravel()
	j = 0
	for i in cols_index:
	  features[i] = temp[j]
	  j += 1 
	result = kidney_model.predict([features])[0] 
	if result == 0:
		return render_template("kidney_dieseas.html", result='Patient have Chonic Kidney Diesease')
	else:
		return render_template("kidney_dieseas.html", result='Cheers, Patient have good Kidneys')

@app.route('/diabetes_predict', methods=['POST', 'GET'])
def predict_diabetes():
	diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
	diabetes_scale_values = pickle.load(open('diabetes_scaler.pkl', 'rb'))

	features = [float(x) for x in request.form.values()]
	features = diabetes_scale_values.transform(np.array(features).reshape(1, -1))
	features = features.ravel()

	result = diabetes_model.predict([features])[0] 
	if result == 0:
		return render_template("diabetes.html", result='Cheers, Patient not have a Diabetes')
	else:
		return render_template("diabetes.html", result='Patient is suffering from Diabetes')

if __name__ == '__main__':
	app.run(debug = True)