from flask import Flask, render_template, request, send_from_directory
import os
import prediction_script

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audioFile' not in request.files:
        return "No audio file found", 400
    
    audio_file = request.files['audioFile']
    if audio_file.filename == '':
        return "No selected file", 400
    
    temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.wav')
    audio_file.save(temp_audio_path)
    
    prediction = prediction_script.predict(temp_audio_path)
    
    return render_template('prediction.html', prediction=prediction)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug= False,host='0.0.0.0')
