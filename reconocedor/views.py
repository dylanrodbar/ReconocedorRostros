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
    
    
    Trozos.cargarImagen(files)
        
    
    return render(request, 'home.html')

def reconocer(request):
    return render(request, 'home.html')
    

    
