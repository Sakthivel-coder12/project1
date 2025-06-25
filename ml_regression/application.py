from flask import Flask, render_template, request
import numpy as np
import pickle

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl','rb'))
scaler_model = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction_datapoint():
    if request.method == 'POST':
        try:
            temp = float(request.form.get('temperature'))
            rh = float(request.form.get('rh'))
            ws = float(request.form.get('ws'))
            rain = float(request.form.get('rain'))
            ffmc = float(request.form.get('ffmc'))
            mc = float(request.form.get('dmc'))
            dc = float(request.form.get('dc'))
            isi = float(request.form.get('isi'))
            bui = float(request.form.get('bui'))
            # classes = float(request.form.get('classes'))
            # region = float(request.form.get('region'))

            input_data = np.array([[temp, rh, ws, rain, ffmc, mc, dc, isi, bui]])
            new_data_scaled = scaler_model.transform(input_data)
            result = ridge_model.predict(new_data_scaled)

            return render_template('prediction.html', results=result[0])
        except Exception as e:
            return render_template('prediction.html', results=f"Error: {str(e)}")
    else:
        return render_template('prediction.html')
if __name__ == "__main__":
    app.run(host='0.0.0.0')
