import cv2
import os
import numpy as np
from numpy import matrix





##
#  Se cargan n imágenes .pmg de dimensiones l * a = p, donde l son la 
#  cantidad de pixeles de largo y a la cantidad de pixeles de ancho,
#  de una carpeta especifica. Posteriormente se convierte cada una
#  en vectores de forma que se guarden en una matriz M de dimensiones
#  p * n. Por último calcula la matriz de covarianza de la matriz M
#  y la guarda en un archivo .txt de nombre MatrizCovarianza.txt en
#  la dirección 'root'.
#  
#  <p>
#  Esta funcion es el metodo principal de la aplicación por lo que
#  se debe de llamar para iniciar la aplicación.
#
#  @param  direccion  la dirección de la carpeta donde se encuentran las imagenes a cargar
#
#  @return      
##

#####
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
    
    
    
##
#  Se convierte una matriz a vector recorriendo en orden sus filas
#
#  @param  matriz  matriz a convertir en vector
#
#  @return vector  vector con los valores de la matriz  
##    
def convertirMatrizAVector(matriz):


    
    if len(matriz) == 1 or len(matriz) == 0:
        return matriz

    else:    
        vector = []
        
        
        for i in matriz:
            for j in i:
                vector.append(j)
        
        return vector


##
#  Se construye una matriz con los vectores de la imagen de muestra
#
#  <p>
#
#  Esta función se utiliza con los vectores creados mediante la funcion
#  convertirMatrizAVector, coloca los vectores de forma vertical en columnas
#  para facilitar su análisis.
#
#  @param  vectores  matriz con los vectores de las imágenes
#
#  @return matriz matriz con los vectores en formas de columna   
##

def crearMatrizDeVectores(vectores):

    if len(vectores) == 1 or len(vectores) == 0:
        return vectores

    else:
        matriz = []
        
        tamanio = len(vectores)
        tamanioV = len(vectores[0])
        
        for i in range(tamanioV):
            matriz.append([])
            for j in range(tamanio):
                matriz[i].append(vectores[j][i])
        
        
        return matriz

##
#  Función la cual calcula la matriz de covarianza de una matriz
#
#
#  @param  matriz  matriz de cualquier dimensión
#
#  @return mCovarianza matriz de covarianza de la matriz ingresada   
## 
def calcularMatrizCovarianza(matriz):
    mCovarianza = np.cov(matriz)
    
    return mCovarianza

##
#  Guarda una matriz en un archivo txt
#
#
#  @param  matriz  matriz de cualquier dimensión
#
#  @return    
## 
def guardarMatrizDeCovarianza(matriz):
    np.savetxt('MatrizCovarianza.txt', matriz)
    
    


##
