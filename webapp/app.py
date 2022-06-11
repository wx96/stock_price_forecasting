from flask import Flask, send_file, make_response, render_template
import os
import datetime
import pandas_datareader.data as web
from plots import plot_df, boxplot, mavg
from IPython.display import Image
import pickle

app = Flask(__name__)

env = os.getenv("run_env", "dev")

if env == "dev":
    BASE_URL = "http://localhost:5000/"
elif env == "prod":
    BASE_URL = "https://python-sci-plotting.herokuapp.com/"


#Get inputs
start = datetime.datetime(2018, 1, 1)
#end = datetime.date.today()
end =  datetime.date.today()

#Amazon: AMZN
#Dayang: 5141.KL
#Genting: 3182.KL
df = web.DataReader("5099.KL", 'yahoo', start, end)


#load model
with open('forecast_model.pckl', 'rb') as fin:
    model = pickle.load(fin)


#API call
@app.route('/')
def index():
    return render_template('index.html', base_url=BASE_URL)

@app.route('/trend')
def plot_trend():
    bytes_obj = plot_df(df, x=df.index, y=df['Adj Close'], title='Stock Prices Across the Year')  
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/boxplot')
def plot_trend_seasonal():
    # Prepare data
    df['year'] = [d.year for d in df.index]
    df['month'] = [d.strftime('%b') for d in df.index]
    years = df['year'].unique()

    bytes_obj = boxplot(df)  
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/mavg')
def plot_mavg():
    bytes_obj = mavg(df)
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

@app.route("/predict", methods=['POST'])
def predict(m2):
    # horizon = int(request.json['horizon'])
    
    future2 = m2.make_future_dataframe(periods=365)
    forecast2 = m2.predict(future2)
    
    data = forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-365:]
    
    ret = data.to_json(orient='records', date_format='iso')
    
    return ret


# @app.route('/plots/breast_cancer_data/pairplot/features/<features>', methods=['GET'])
# def pairplot(features):
#     try:
#         # parse columns
#         parsed_features = [feature.strip() for feature in features.split(',')]
#         bytes_obj = get_pair_plot_as_bytes(breast_cancer_df, parsed_features)

#         return send_file(bytes_obj,
#                          attachment_filename='plot.png',
#                          mimetype='image/png')
#     except ValueError:
#         # something went wrong to return bad request
#         return make_response('Unsupported request, probably feature names are wrong', 400)


# @app.route('/plots/breast_cancer_data/correlation_matrix', methods=['GET'])
# def correlation_matrix():
#     bytes_obj = get_correlation_matrix_as_bytes(breast_cancer_df, features_names)

    # return send_file(bytes_obj,
    #                  attachment_filename='plot.png',
    #                  mimetype='image/png')


if __name__ == '__main__':
    if env == "dev":
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.run(debug=True)
    else:
        app.run(debug=False)
