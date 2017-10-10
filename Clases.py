import cv2
import numpy as np


class Entrenamiento:

    def __init__(self):
        self.matrizMuestras = []
        self. centroides = []
        self.diffPrima = []
        self.eigen = []
        self.cara_promedio = []
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
#  @param  files  archivos subidos desde Django
#
#  @return
##

    def cargarImagen(self, files):
        vectores = []
        num_sujetos = 0
        for i in files:
            imagen = i.read()
            img = cv2.imdecode(np.fromstring(imagen, np.uint8), -1)
            filename = i.name
            sujeto = filename.split("-")[0]
            if(int(sujeto) > num_sujetos):
                num_sujetos = int(sujeto)
            vectores.append([int(sujeto), self.convertirMatrizAVector(img)])
        sujetos = [0]*num_sujetos
        for i in vectores:
            if(sujetos[i[0]-1] == 0):
                sujetos[i[0]-1] = [i[1]]
            else:
                sujetos[i[0]-1] += [i[1]]
        vectores = []
        for i in sujetos:
            for j in i:
                vectores.append(j)

        self.matriz = self.crearMatrizDeVectores(vectores)

        self.matriz = np.array(self.matriz)

        self.cara_promedio = self.cara_prom(self.matriz)

        diferencias = self.matriz_diferencias(self.matriz, self.cara_promedio)

        self.eigen = self.auto_vectores(diferencias, 100)

        self.guardarMatrizTxt(self.eigen[1], "autovectores.txt")

        self.diffPrima = np.matmul(np.transpose(self.eigen[1]), diferencias)

        self.guardarMatrizTxt(self.diffPrima, "autocaras.txt")

        self.centroides = self.calcular_centroides(self.diffPrima, 10)

        self.guardarMatrizTxt(self.centroides, "centroides.txt")
##
#  Se convierte una matriz a vector recorriendo en orden sus filas
#
#  @param  matriz  matriz a convertir en vector
#
#  @return vector  vector con los valores de la matriz
##

    def convertirMatrizAVector(self, matriz):
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

    def crearMatrizDeVectores(self, vectores):

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
#  Guarda una matriz en un archivo txt
#
#
#  @param  matriz  matriz de cualquier dimensión
#
#  @return
##

    def guardarMatrizTxt(self, matriz, nombre):
        np.savetxt(nombre, matriz)

##
#  Calcula el promedio de los valores de un vector
#
#
#  @param  vectores  matriz con los vectores de entrenamiento
#                    del tipo numpy.array
#
#  @return prom    promedio de todos los vectores, en este caso "cara promedio"
##

    def cara_prom(self, vectores):
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
#  @param  vectores  matriz con los vectores de entrenamiento del
#                    tipo numpy.array
#  @param  promedio  promedio de todos los vectores
#
#  @return diff     la matriz de diferencias
##

    def matriz_diferencias(self, vectores, promedio):
        diff = []
        for i in range(0, len(vectores[0])):
            diff += [vectores[:, i] - promedio]
        return np.transpose(diff)

    def normalizar(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

##
#  Calcula los autovectores de una matriz de covarianza
#  dada una matriz de diferencias
#
#
#  @param  matriz_diferencias  matriz con la diferencia entre
#          los vectores y el promedio
#
#  @return [auto_valores,auto_vectores] arreglo con dos posiciones,
#                                       auto valores y autovectores
##

    def auto_vectores(self, matriz_diferencias):
        covarianzaV = np.matmul(np.transpose(matriz_diferencias),
                                matriz_diferencias)
        self.eigen = np.linalg.eig(covarianzaV)
        auto_valores = self.eigen[0]
        auto_vectoresV = self.eigen[1]
        auto_vectores = np.matmul(matriz_diferencias, auto_vectoresV)
        auto_vectoresNorm = []
        for i in range(0, len(auto_vectores[0])):
            auto_vectoresNorm += [self.normalizar(auto_vectores[:, i])]
        return [auto_valores, np.transpose(np.array(auto_vectoresNorm))]
##
#  Calcula los centroides de un arreglo de vectores
#
#
#  @param  auto_caras  matriz con las auto_caras a calcular el centroide
#
#  @return centroides arreglo del tipo numpy
##

    def calcular_centroides(self, autocaras, num_muestras):
        self.centroides = []
        j = 0

        for i in range(0, int(len(autocaras[0])/num_muestras)):

            self.centroides += [self.cara_prom(
                autocaras[:, j:j + num_muestras])]
            j += num_muestras
            i = i
        return np.transpose(np.array(self.centroides))


class Reconocedor:

    def __init__(self, entrenamiento):
        self.entrenamiento = Entrenamiento()

##
#  Calcula la distancia entre dos puntos en un espacio euclidiano
#
#
#  @param  punto1  vector con puntos unidimensionales
#  @param  punto2  vector con puntos unidimensionales
#
#  @return distancia distancia entre los puntos
##

    def distancia_euclidiana(self, punto1, punto2):
        if (len(punto1) != len(punto2)):
            raise ValueError("Deben tener la misma dimensionalidad")
        else:
            distancia = 0
            for i in range(0, len(punto1)):
                distancia += (punto1[i] - punto2[i]) ** 2
            return np.sqrt(distancia)

##
#  Retorna la etiqueta correspondiente al sujeto de la imagen que se provea
#
#
#  @param  archivo  dirección de la imagen
#
#  @return etiqueta etiqueta del sujeto correspondiente
##

    def reconocer(self, archivo):
        imagen = cv2.imread(archivo, 0)
        vector = self.entrenamiento.convertirMatrizAVector(imagen)
        etiqueta = 0
        menor_dist = 0

        vector = np.array(vector) - self.entrenamiento.cara_promedio

        vector = np.matmul(np.transpose(self.entrenamiento.autovectores),
                           vector)

        for i in range(0, len(self.entrenamiento.centroides[0])):
            distancia = self.distancia_euclidiana(
                vector, self.entrenamiento.centroides[:, i])
            if(i == 0):
                menor_dist = distancia
                etiqueta = i
            else:
                if(distancia < menor_dist):
                    menor_dist = distancia
                    etiqueta = i
        return etiqueta + 1
