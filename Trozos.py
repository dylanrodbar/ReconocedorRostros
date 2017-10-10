import cv2
import os
import numpy as np
from numpy import matrix
from builtins import str
from reconocedor.Trozos import auto_vectores

centroides = [] #atributo de la clase de entrenamiento, en el init debería intentar cargar el archivo centroides.txt
diffPrima = [] #atributo de la clase de entrenamiento, en el init debería intentar cargar el archivo diffprima.txt
eigen = []
cara_promedio = []

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
##
#####
##
def cargarImagen(files):
    vectores = []
    num_sujetos = 0
    for i in files:
        imagen = i.read()
        img = cv2.imdecode(np.fromstring(imagen, np.uint8), -1)
        filename = i.name
        sujeto = filename.split("-")[0]
        if(int(sujeto) > num_sujetos):
            num_sujetos = int(sujeto)
        vectores.append([int(sujeto),convertirMatrizAVector(img)])
    sujetos = [0]*num_sujetos
    for i in vectores:
        if(sujetos[i[0]-1] == 0 ):
            sujetos[i[0]-1] = [i[1]]
        else:
            sujetos[i[0]-1] += [i[1]]
    vectores = []
    for i in sujetos:
           for j in i:
               vectores.append(j)
    
    matriz = crearMatrizDeVectores(vectores)
    
    
    #mCovarianza = Trozos.calcularMatrizCovarianza(matriz)
    
    #Trozos.guardarMatrizTxt(mCovarianza, "Matriz")
    matriz = np.array(matriz)
    global cara_promedio
    cara_promedio= cara_prom(matriz)

    diferencias = matriz_diferencias(matriz,cara_promedio)
    global eigen
    eigen = auto_vectores(diferencias, 100)
    guardarMatrizTxt(eigen[1], "autovectores.txt")
    global diffPrima
    diffPrima = np.matmul(np.transpose(eigen[1]),diferencias)
    
    guardarMatrizTxt(diffPrima,"autocaras.txt")
    global centroides  
    centroides= calcular_centroides(diffPrima,10)

    guardarMatrizTxt(centroides,"centroides.txt")
    
    print("Reconociendo")
    
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-1.pgm"))
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-2.pgm"))
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-3.pgm"))
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-4.pgm"))
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-5.pgm"))
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-6.pgm"))
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-7.pgm"))
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-8.pgm"))
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-9.pgm"))
    print(reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s25\\25-10.pgm"))
   
    
        
    
    
    
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
def guardarMatrizTxt(matriz,nombre):
    np.savetxt(nombre, matriz)
    
    
##
#  Calcula el promedio de los valores de un vector
#
#
#  @param  vectores  matriz con los vectores de entrenamiento del tipo numpy.array
#
#  @return prom    promedio de todos los vectores, en este caso "cara promedio"
## 
def cara_prom(vectores):
    prom = np.zeros(len(vectores))
  
    #vectores = np.array(vectores) la matriz de vectores debe ser del tipo numpy.array
    for i in range(0,len(vectores[0])):
        vector = vectores[:,i]
      
 
        
        prom += vector
    prom = prom / len(vectores[0])
  
    return prom

##
#  Calcula la matriz de diferencias de n vectores de la forma:
#  di = vi - promedio, donde di es el vector de diferencia i,
#  vi el vector i y promedio el promedio generado con la funcion
#  cara_prom.
#
#
#  @param  vectores  matriz con los vectores de entrenamiento del tipo numpy.array
#  @param  promedio  promedio de todos los vectores
#
#  @return diff     la matriz de diferencias
## 
def matriz_diferencias(vectores,promedio):
    diff= []
    for i in range(0,len(vectores[0])):
        diff += [vectores[:,i]-promedio]
    return np.transpose(diff)


def normalizar(vector):
    norm=np.linalg.norm(vector)
    if norm==0: 
       return vector
    return vector/norm
##
#  Calcula los autovectores de una matriz de covarianza dada una matriz de diferencias
#
#
#  @param  matriz_diferencias  matriz con la diferencia entre los vectores y el promedio
#
#  @return [auto_valores,auto_vectores] arreglo con dos posiciones, auto valores y autovectores
## 
def auto_vectores(matriz_diferencias, porcentaje_autovectores):
    covarianzaV = np.matmul(np.transpose(matriz_diferencias),matriz_diferencias)
    eigen = np.linalg.eig(covarianzaV)
    auto_valores = eigen[0]
    auto_vectoresV = eigen[1]
    auto_vectores = np.matmul(matriz_diferencias,auto_vectoresV)#cambiar, usar el atributo de la clase
    idx = auto_valores.argsort()[::-1]  
    auto_vectores = auto_vectores[:,idx]
    num_autovect = len(auto_valores)
    num_autovect = int(num_autovect * (porcentaje_autovectores/100))
    auto_vectores = auto_vectores[:,0:num_autovect]
    auto_vectoresNorm = []
    for i in range(0,len(auto_vectores[0])):
        auto_vectoresNorm += [normalizar(auto_vectores[:,i])]
    
    
    
    
    return [auto_valores,np.transpose(np.array(auto_vectoresNorm))]
##
#  Calcula los centroides de un arreglo de vectores
#
#
#  @param  auto_caras  matriz con las auto_caras a calcular el centroide
#
#  @return centroides arreglo del tipo numpy 
## 
def calcular_centroides(autocaras,num_muestras):
    centroides= []
    j=0
   
    for i in range(0,int(len(autocaras[0])/num_muestras)):
       
        centroides += [cara_prom(autocaras[:,j:j +num_muestras])]
        j+=num_muestras

    
    return np.transpose(np.array(centroides))
##
#  Calcula la distancia entre dos puntos en un espacio euclidiano
#
#
#  @param  punto1  vector con puntos unidimensionales
#  @param  punto2  vector con puntos unidimensionales
#
#  @return distancia distancia entre los puntos 
##
def distancia_euclidiana(punto1,punto2):
    if (len(punto1) != len(punto2)):
        raise ValueError("Los puntos a calcular su distancia deben tener la misma dimensionalidad")
    else:
        distancia = 0
        for i in range(0,len(punto1)):
            distancia += (punto1[i]-punto2[i])**2
        return np.sqrt(distancia)
##
#  Retorna la etiqueta correspondiente al sujeto de la imagen que se provea
#
#
#  @param  archivo  dirección de la imagen
#
#  @return etiqueta etiqueta del sujeto correspondiente 
##    
def reconocer(archivo):
    imagen = cv2.imread(archivo, 0)
    #img = cv2.imdecode(np.fromstring(imagen, np.uint8), -1)
    
    vector = convertirMatrizAVector(imagen)
    #print(vector)
    etiqueta = 0
    menor_dist = 0
    global cara_promedio
    vector = np.array(vector) - cara_promedio
    global eigen
    vector = np.matmul(np.transpose(eigen[1]),vector)
    global centroides #cambiar, usar el atributo de la clase
    
    
    for i in range(0,len(centroides[0])):
        distancia = distancia_euclidiana(vector,centroides[:,i])
        
        if(i==0):
            menor_dist = distancia
            etiqueta = i
        else:
            if(distancia < menor_dist):
                menor_dist = distancia
                etiqueta = i
    
    return etiqueta + 1
        
        


