from django.shortcuts import render
import cv2
import os
import numpy as np
from numpy import matrix
import Trozos

# Create your views here.




def inicio(request):
    return render(request, 'home.html')
    
def procesar(request):
    vectores = []
    files = request.FILES.getlist("file")
    
    
    print(files)
        
    for i in files:
        print("******************************")
        imagen = i.read()
        img = cv2.imdecode(np.fromstring(imagen, np.uint8), -1)
        vectores.append(Trozos.convertirMatrizAVector(img))
        print(img)  
        print("******************************")
        
    matriz = Trozos.crearMatrizDeVectores(vectores)
    
    
    mCovarianza = Trozos.calcularMatrizCovarianza(matriz)
    
    Trozos.guardarMatrizDeCovarianza(mCovarianza)
    
        
    return render(request, 'home.html')

    
