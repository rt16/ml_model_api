# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from flask_cors import CORS, cross_origin

# Your API definition 
app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/predict', methods=['GET'])
@cross_origin(supports_credentials=True)
def predict():
    
        try:            
            return 'Hello World'
        

        except:

            return jsonify({'trace': traceback.format_exc()})
   

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)
