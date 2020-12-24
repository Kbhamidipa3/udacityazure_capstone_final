#writefile score.py
import pickle
import json
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path(model_name = "lv-automl-classification")
    model = joblib.load(model_path)

def run(rawdata):
    try:
        data = json.loads(rawdata)['data']
        result = model.predict(pd.DataFrame(data, columns = ['Score']))
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    return json.dumps({"result": result.tolist()})