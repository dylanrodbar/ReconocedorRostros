from django.conf.urls import url, include
from . import views


urlpatterns = [
    url(r'^reco/$', views.inicio, name='inicio'),
    url(r'^recoProcesar/$', views.procesar, name='procesar'),
 
]