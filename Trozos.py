import cv2
import random
import numpy as np
import csv
from time import time

from reconocedor.Trozos import auto_vectores

centroides = []
diffPrima = []
eigen = []
cara_promedio = []

##
#  Se cargan n im谩genes .pmg de dimensiones l * a = p, donde l son la
#  cantidad de pixeles de largo y a la cantidad de pixeles de ancho,
#  de una carpeta especifica. Posteriormente se convierte cada una
#  en vectores de forma que se guarden en una matriz M de dimensiones
#  p * n. Por 煤ltimo calcula la matriz de covarianza de la matriz M
#  y la guarda en un archivo .txt de nombre MatrizCovarianza.txt en
#  la direcci贸n 'root'.
#
#  <p>
#  Esta funcion es el metodo principal de la aplicaci贸n por lo que
#  se debe de llamar para iniciar la aplicaci贸n.
#
#  @param  direccion  la direcci贸n de la carpeta donde se encuentran las
#                     imagenes a cargar
#
#  @return
##


def cargarImagen(files):
    vectores = []
    num_sujetos = 0
    tiempo_entrenamiento = time()
    for i in files:
        imagen = i.read()
        img = cv2.imdecode(np.fromstring(imagen, np.uint8), -1)
        filename = i.name
        sujeto = filename.split("-")[0]
        if(int(sujeto) > num_sujetos):
            num_sujetos = int(sujeto)
        vectores.append([int(sujeto), convertirMatrizAVector(img)])
    sujetos = [0]*num_sujetos
    for i in vectores:
        if(sujetos[i[0]-1] == 0):
            sujetos[i[0]-1] = [i[1]]
        else:
            sujetos[i[0]-1] += [i[1]]
    vectores = []
    muestras_reconocer = []
    for i in sujetos:
        recon = random.sample(range(1, 11), 2)
        muestras_reconocer += [recon]
        cont = 1
        for j in i:
            if cont not in recon:
                vectores.append(j)
            cont += 1

    matriz = crearMatrizDeVectores(vectores)

    matriz = np.array(matriz)
    global cara_promedio
    cara_promedio = cara_prom(matriz)

    diferencias = matriz_diferencias(matriz, cara_promedio)
    global eigen
    eigen = auto_vectores(diferencias, 10)
    guardarMatrizTxt(eigen[1], "autovectores.txt")
    global diffPrima
    diffPrima = np.matmul(np.transpose(eigen[1]), diferencias)

    guardarMatrizTxt(diffPrima, "autocaras.txt")
    global centroides
    centroides = calcular_centroides(diffPrima, 8)

    guardarMatrizTxt(centroides, "centroides.txt")
    tiempo_entrenamiento = time() - tiempo_entrenamiento

    print("Reconociendo")
    matriz_confusion = [[0 for i in range(41)] for j in range(41)]
    tiempos = []
    for i in range(1, 42):
        print("reconociendo "+str(i))
        etiquet = []

        tiempoInicio = time()
        etiquet += [reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s"
                              + str(i) + "\\" + str(i) + "-"
                              + str(muestras_reconocer[i-1][0]) + ".pgm")]
        tiempo = time() - tiempoInicio
        tiempos += [tiempo]
        tiempoInicio = time()
        etiquet += [reconocer("C:\\Users\\josem\\Desktop\\PCA\\input\\s"
                              + str(i) + "\\" + str(i) + "-"
                              + str(muestras_reconocer[i-1][1]) + ".pgm")]
        tiempo = time() - tiempoInicio
        tiempos += [tiempo]
        print(etiquet)
        if(etiquet[0] == i):
            matriz_confusion[i-1][i-1] += 1
        else:
            matriz_confusion[etiquet[0]-1][i-1] += 1
        if(etiquet[1] == i):
            matriz_confusion[i-1][i-1] += 1
        else:
            matriz_confusion[etiquet[1]-1][i-1] += 1

    precisiones = []
    recalls = []
    csvfile = open('metricas.csv', 'w+')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Sujeto ', 'VP', 'FP', 'FN', 'Precision', 'Recall'])
    for i in range(1, 42):
        metric = metricas(matriz_confusion, i)
        precision = 0
        recall = 0
        if(metric[0]+metric[1] != 0):
            precision = metric[0] / (metric[0] + metric[1])

        precisiones += [precision]
        if(metric[0] + metric[2] != 0):
            recalls = metric[0] / (metric[0] + metric[2])
        recalls += [recall]
        writer.writerow([str(i), str(metric[0]), str(metric[1]),
                        str(metric[2]), precision, recall])
    writer.writerow(['Precision', np.average(precisiones)])
    writer.writerow(['Recall', np.average(recalls)])
    print(np.average(precisiones))
    print(np.average(recalls))
    print("Tiempo de entrenamiento: " + str(tiempo_entrenamiento))
    print("Tiempo promedio de reconocimiento: " + str(np.average(tiempos)))
    csvfile.close()

##
#  Funci髇 la cual calcula los Verdaderos Positivos,
#  Falsos Positivos y Falsos Negativos dada una
#  matriz de confusi髇 y el n鷐ero de la clase
#  a calcular dichas m閠ricas.
#
#  @param  matriz  matriz de confusi髇
#  @param  sujeto  n鷐ero de clase(entero)
#
#  @return [VP,FP,FN]  vector con los Verdaderos Positivos
#                      Falsos positivos y Falsos Negativos
##


def metricas(matriz, sujeto):
    VP = matriz[sujeto-1][sujeto-1]
    matriz = np.array(matriz)
    FP = np.sum(matriz[sujeto-1][np.arange(len(matriz[sujeto - 1]))
                                 != sujeto - 1])
    FN = np.sum(matriz[:, sujeto - 1])
    FN = FN - matriz[:, sujeto - 1][sujeto - 1]
    return [VP, FP, FN]

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
#  Esta funci贸n se utiliza con los vectores creados mediante la funcion
#  convertirMatrizAVector, coloca los vectores de forma vertical en columnas
#  para facilitar su an谩lisis.
#
#  @param  vectores  matriz con los vectores de las im谩genes
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
#  Funci贸n la cual calcula la matriz de covarianza de una matriz
#
#
#  @param  matriz  matriz de cualquier dimensi贸n
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
#  @param  matriz  matriz de cualquier dimensi贸n
#
#  @return
##


def guardarMatrizTxt(matriz, nombre):
    np.savetxt(nombre, matriz)

##
#  Calcula el promedio de los valores de un vector
#
#
#  @param  vectores  matriz con los vectores de entrenamiento del
#                    tipo numpy.array
#
#  @return prom    promedio de todos los vectores, en este caso "cara promedio"
##


def cara_prom(vectores):
    prom = np.zeros(len(vectores))

    for i in range(0, len(vectores[0])):
        vector = vectores[:, i]
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
#  @param  vectores  matriz con los vectores de
#                    entrenamiento del tipo numpy.array
#  @param  promedio  promedio de todos los vectores
#
#  @return diff     la matriz de diferencias
##


def matriz_diferencias(vectores, promedio):
    diff = []
    for i in range(0, len(vectores[0])):
        diff += [vectores[:, i] - promedio]
    return np.transpose(diff)

##
#  Dado un vector, se calcula su norma para poder normalizarlo,
#  lo cual consiste en que su magnitud sea igual a 1.
#
#  @param  vector  vector del tipo numpy.array()
#
#  @return vector/norm  vector normalizado
##


def normalizar(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

##
#  Calcula los autovectores de una matriz de
#  covarianza dada una matriz de diferencias
#
#
#  @param  matriz_diferencias  matriz con la diferencia
#                              entre los vectores y el promedio
#
#  @return [auto_valores,auto_vectores] arreglo con dos
#                                       posiciones, auto valores y autovectores
##


def auto_vectores(matriz_diferencias, porcentaje_autovectores):
    covarianzaV = np.matmul(np.transpose(matriz_diferencias),
                            matriz_diferencias)
    eigen = np.linalg.eigh(covarianzaV)
    auto_valores = eigen[0]
    auto_vectoresV = eigen[1]
    auto_vectores = np.matmul(matriz_diferencias, auto_vectoresV)
    num_autovect = len(auto_valores)
    print(num_autovect)
    num_autovect = int(num_autovect * (porcentaje_autovectores/100))
    print(num_autovect)
    auto_vectores = auto_vectores[:, -num_autovect:]
    auto_vectoresNorm = []
    for i in range(0, len(auto_vectores[0])):
        auto_vectoresNorm += [normalizar(auto_vectores[:, i])]
    return [auto_valores, np.transpose(np.array(auto_vectoresNorm))]

##
#  Calcula los centroides de un arreglo de vectores
#
#
#  @param  auto_caras  matriz con las auto_caras a calcular el centroide
#
#  @return centroides arreglo del tipo numpy
##


def calcular_centroides(autocaras, num_muestras):
    centroides = []
    j = 0
    for i in range(0, int(len(autocaras[0])/num_muestras)):
        centroides += [cara_prom(autocaras[:, j:j + num_muestras])]
        j += num_muestras
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


def distancia_euclidiana(punto1, punto2):
    if (len(punto1) != len(punto2)):
        raise ValueError(
            "Los puntos a calcular su distancia deben tener" +
            "la misma dimensionalidad")
    else:
        distancia = 0
        for i in range(0, len(punto1)):
            distancia += (punto1[i] - punto2[i]) ** 2
        return np.sqrt(distancia)

##
#  Retorna la etiqueta correspondiente al sujeto de la imagen que se provea
#
#
#  @param  archivo  direcci贸n de la imagen
#
#  @return etiqueta etiqueta del sujeto correspondiente
##


def reconocer(archivo):
    imagen = cv2.imread(archivo, 0)
    vector = convertirMatrizAVector(imagen)
    etiqueta = 0
    menor_dist = 0
    global cara_promedio
    vector = np.array(vector) - cara_promedio
    global eigen
    vector = np.matmul(np.transpose(eigen[1]), vector)
    global centroides
    for i in range(0, len(centroides[0])):
        distancia = distancia_euclidiana(vector, centroides[:, i])
        if(i == 0):
            menor_dist = distancia
            etiqueta = i
        else:
            if(distancia < menor_dist):
                menor_dist = distancia
                etiqueta = i

    return etiqueta + 1
