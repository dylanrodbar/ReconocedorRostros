import cv2
import os
import numpy as np
from numpy import matrix





########### Primer trozo: lectura de imagenes de una carpeta ###########

def cargarImagen(direccion):
    vectores = []
    
    for archivo in os.listdir(direccion): #Se recorren los archivos dentro de la carpeta
        imagen = cv2.imread(os.path.join(direccion, archivo), -1) #Se abre un archivo pmg
        
        
        ########### Segundo trozo: convertir la matriz a vector ###########

        if imagen is not None: #Si se abre exitosamente (existe)
            vectores.append(convertirMatrizAVector(imagen)) #Se pasa la matriz a vector
            
    
    
    matriz = crearMatrizDeVectores(vectores)
    
    
    mCovarianza = calcularMatrizCovarianza(matriz)
    
    guardarMatrizDeCovarianza(mCovarianza)
    
    
    
    
def convertirMatrizAVector(matriz):
        
    vector = []
    
    
    for i in matriz:
        for j in i:
            vector.append(j)
    
    return vector


########### Tercer trozo: hacer una matriz con los vectores de la imagen de muestra  ###########


def crearMatrizDeVectores(vectores):
    matriz = []
    
    tamanio = len(vectores)
    tamanioV = len(vectores[0])
    
    for i in range(tamanioV):
        matriz.append([])
        for j in range(tamanio):
            matriz[i].append(vectores[j][i])
    
    
    return matriz

########### Cuarto trozo: calcular la matriz de covarianza de la matriz que sale del punto anterior ########### 
def calcularMatrizCovarianza(matriz):
    mCovarianza = np.cov(matriz)
    
    for i in mCovarianza:
        print(i)
        break
    
    return mCovarianza


def guardarMatrizDeCovarianza(matriz):
    np.savetxt('MatrizCovarianza.txt', matriz)
    
    
    ###
#crearMatrizDeVectores([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]], [[13,14,15], [16,17,18]]])
#cargarImagen('input/s1') 
#convertirMatrizAVector([[1,2,3],[4,5,6],[7,8,9]])
#



##
