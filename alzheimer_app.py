import numpy as np
import os
from keras.preprocessing import image
import tensorflow as tf
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import tensorflow.compat.v1 as tf
from tensorflow.keras.applications.xception import preprocess_input
global graph,text,prediction
tf.compat.v1.disable_eager_execution()
session=tf.compat.v1.Session()
tf.disable_v2_behavior()
graph=tf.get_default_graph()
app=Flask(__name__)
set_session(session)
model=load_model('alzheimer_3.h5',compile=False)
def predict(text):
	return render_template("result.html",text=text)
@app.route('/')
def index():
	return render_template("AlMLProject.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
	if request.method == 'POST':
		text='NA'
		f=request.files['mri-scan']
		f_name=secure_filename(f.filename)
		basepath = os.path.dirname(os.path.abspath(__file__))
		file_path=os.path.join(basepath,f_name)
		f.save(file_path)
		img=image.load_img(file_path,target_size=(180,180))
		x=image.img_to_array(img)
		x=np.expand_dims(x,axis=0)
		with graph.as_default():
			set_session(session)
			prediction=np.argmax(model.predict(x),axis=1)[0]
		if(prediction==0):
			text = 'Mild Demented'
		if(prediction==1):
			text = 'Moderate Demented'
		if(prediction==2):
			text='Non Demented'
		if(prediction==3):
			text='Very Mildly Demented'
		return predict(text)
if(__name__=="__main__"):
	app.run(debug=True)

