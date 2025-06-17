from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import os
import pickle
import tensorflow as tf

def project_financial_risk_main(request):

    return render(request, "project_financial_risk/project_financial_risk_main.html")


@csrf_exempt
def predict_financial_risk(request):

    FEATURE_NAMES = [

    "CreditScore",

    "AnnualIncome",

    "LoanAmount",

    "EmergencyFundBalance",

    "CreditCardUtilizationRate",

    "TotalLiabilities",

    ]

    MODEL_PATH = os.path.join( settings.BASE_DIR, 'project_financial_risk', 'machine_learning_financial_risk', 'financial_risk_model.h5' )

    SCALER_PATH = os.path.join( settings.BASE_DIR, 'project_financial_risk', 'machine_learning_financial_risk', 'standard_scaler.pkl' )


    model = tf.keras.models.load_model(MODEL_PATH)

    with open(SCALER_PATH, 'rb') as f:
        
        scaler = pickle.load(f)

    data = json.loads(request.body)

    input_features = [data.get(feature) for feature in FEATURE_NAMES]
            
    input_array = np.array(input_features).reshape(1, -1)

    input_scaled = scaler.transform(input_array)

    prediction_prob = model.predict(input_scaled)[0][0]

    prediction = int(prediction_prob >= 0.5)
            
    return JsonResponse({ "loan_approved": prediction, "approval_probability": float(prediction_prob) })