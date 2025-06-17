from django.urls import path
from . import views

urlpatterns = [

    path('main', views.project_financial_risk_main, name='project-financial-risk-main'),
    
    path('api/financial_risk', views.predict_financial_risk, name='predict-financial-risk')

]