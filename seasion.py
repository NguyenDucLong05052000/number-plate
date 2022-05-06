from flask import Flask,redirect,url_for,render_template,request,session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os 
from os import path
from demo import Full_module
import mysql.connector
app = Flask(__name__)
mydb = mysql.connector.connect(
    host='mysql',
    user='root',
    passwd='1234',port=3306,
)
#auth_plugin='mysql_native_password'
#host='localhost'
cur = mydb.cursor()
CORS(app)
app.config['CORS_HEADERS']= 'Content-Type'

UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("gui1.html")

@app.route('/', methods=['POST'])
def upload_image():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    a=app.config['UPLOAD_FOLDER']+ filename
    result = Full_module(a)
    link_image = UPLOAD_FOLDER+filename
    numberplate = result
    cur.execute("use users")
    cur.execute("INSERT INTO numberplate_id(link_image,numberplate) VALUES(%s,%s)",(link_image, numberplate))
    mydb.commit()
    return render_template('gui1.html',content=str(result),filename = filename)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__=="__main__":    
    app.run(host='0.0.0.0',port='8885',debug=True) 