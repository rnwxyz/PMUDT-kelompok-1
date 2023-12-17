from flask import Flask, request, jsonify, render_template, flash, redirect, session, url_for
from database import db
from config import Config
from service.userService import UserService
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from model.userModel import User
from service.analysisService import AnalysisService
import pandas as pd
import numpy as np

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost:3306/flask'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = Config.SECRET_KEY

db = db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/', methods=['GET'])
@login_required
def home(): 
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login(): 
    # Handle login form submission
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = UserService.login(email, password)
        if user is None:
            return render_template('login.html', error='email atau password salah')
        login_user(user)
        return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register(): 
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        password_confirm = request.form['password_confirm']
        email = request.form['email']

        if password != password_confirm:
            return render_template('register.html', error='konfirmasi password tidak sesuai')
        
        # cek email sudah terdaftar atau belum
        user = UserService.getUser(email)
        if user is not None:
            return render_template('register.html', error='email sudah terdaftar')
        
        UserService.createUser(name, password, email)
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/analysis', methods=['GET', 'POST'])
@login_required
def analysis(): 
    if request.method == 'POST':
        document = np.array([])

        # get text data
        text_data = request.form['document']
        if text_data:
            text_data = np.array(text_data.splitlines())
            document = np.concatenate((document, text_data), axis=None)
        
        # get file data
        file = request.files['file']
        if file:
            try:
                # Read the CSV file using Pandas
                csv_data = pd.read_csv(file)

                # Process the data as needed convert to numpy array
                file_doc = csv_data.to_numpy()
                document = np.concatenate((document, file_doc), axis=None)

            except pd.errors.EmptyDataError:
                return render_template('analysis.html', error='data tidak ditemukan')
            except pd.errors.ParserError:
                return render_template('analysis.html', error='data tidak ditemukan')
        
        preprocessed, error = AnalysisService.preprocess(document)
        if error:
            return render_template('analysis.html', error=error)
        
        result, error = AnalysisService.predict(preprocessed)
        if error:
            return render_template('analysis.html', error=error)
        
        error = AnalysisService.create_plot(result)
        if error:
            return render_template('analysis.html', error=error)
        
        # form pandas dataframe to json
        result = result.to_json(orient='records')
        # redirect to result page
        session['result'] = result
        return redirect(url_for('result'))

    return render_template('analysis.html')

@app.route('/history')
@login_required
def history(): 
    return render_template('history.html')

@app.route('/result', methods=['GET', 'POST'])
@login_required
def result(): 
    # get result data
    result = session.get('result', None)
    
    return render_template('analysisResult.html', result=result)

@app.route('/logout')
@login_required
def logout(): 
    logout_user()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=False)