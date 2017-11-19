import unittest
import Trozos
import numpy as np
from _overlapped import NULL

##
#  Clase compuesta por las pruebas unitarias del archivo Trozos.py 
#  
#
#  @param  TestCase Caso de uso a probar
#    
##

class TestMyModule(unittest.TestCase):
    
#Pruebas unitarias

##
#  Prueba unitaria para la funciÃ³n convertirMatrizAVector
#  Prueba tres casos de uso:
#       1. Una matriz de 2x3
#       2. Una matriz vacia
#       3. Una vector con cero
#  @param
#
#  @return
##     

    def test_convertirMatrizAVector(self):
        self.assertEqual(Trozos.convertirMatrizAVector(
            [[1, 2, 3], [4, 5, 6]]), [1, 2, 3, 4, 5, 6])
        self.assertEqual(Trozos.convertirMatrizAVector([]), [])
        self.assertEqual(Trozos.convertirMatrizAVector([0]), [0])
        
##
#  Prueba unitaria para la funciÃ³n crearMatrizDeVectores.
#  Prueba tres casos de uso:
#       1. Una matriz de 2x4
#       2. Una matriz vacia
#       3. Una matriz 1x1 con 1
#  @param
#
#  @return
##        
    def test_crearMatrizDeVectores(self):
        self.assertEqual(Trozos.crearMatrizDeVectores(
            [[1, 2, 3, 4], [5, 6, 7, 8]]), [[1, 5], [2, 6], [3, 7], [4, 8]])
        self.assertEqual(Trozos.crearMatrizDeVectores([]), [])
        self.assertEqual(Trozos.crearMatrizDeVectores([[1]]), [[1]])
##
#  Prueba unitaria para la funciÃ³n calcularMatrizCovarianza.
#  Prueba como entrada una matriz 1x4.
#    
##        
    def test_calcularMatrizCovarianza(self): 
        self.assertEqual(Trozos.calcularMatrizCovarianza(
            [[1, 2, 3, 4]]), np.array(1.6666666666666665))  
    
##
#  Prueba unitaria para la función normalizar
#  Prueba como entrada un vector 1xn
#  @param
#
#  @return
##

    def test_normalizar(self):
        self.assertAlmostEqual(np.linalg.norm(Trozos.normalizar(
            [123.0032, 123.4, 36.0043, 3, 0, 345.19])), 1, None, 0.1)
##
#  Prueba unitaria para la funcion cara_prom.
#  Prueba dos casos de uso:
#       1. Una matriz de 2x3
#       2. Una matriz 2x4 con ceros
#
#  @param
#
#  @return
##

    def test_cara_prom(self):
        
        self.assertEqual(Trozos.cara_prom(
            np.array([[11, 10, 4], [100, 200, 20]])).tolist(), [25/3, 320/3])
        self.assertEqual(Trozos.cara_prom(
            np.array([[0, 0, 0, 0], [0, 0, 0, 0]])).tolist(), [0, 0])
    
#Pruebas de integración

##
#  Prueba de integración entre las funciones convertirMatrizAVector
#  y crearMatrizDeVectores
#
#
#  @param Matriz1 matriz con numeros
#  @param Matrizn Hasta N numeros de matriz de entradas a probar
#
#  @return true
##

    def test_MatrizVectores(self):
        matriz1 = [[1, 2, 3], [0, 0, 0],
                    [-0.23, 0.3, -9]]
        matriz2 = [[10, 223, 43], [10, 10, 10],
                    [-0.23234, 0.35, -69]]
        matriz3 = [[0, 0, 0], [0, 0, 0],
                    [0, 0, 0]]
        vectores = []
        vectores += [Trozos.convertirMatrizAVector(matriz1)]
        vectores += [Trozos.convertirMatrizAVector(matriz2)]
        vectores += [Trozos.convertirMatrizAVector(matriz3)]
        
        self.assertEqual(Trozos.crearMatrizDeVectores(vectores),
                         [[1, 10, 0], [2, 223, 0], [3, 43, 0], [0, 10, 0],
                          [0, 10, 0], [0, 10, 0], [-0.23, -0.23234, 0],
                          [0.3, 0.35, 0],[-9, -69, 0]])
##
#  Prueba de integración entre las funciones crearMatrizDeVectores
#  y cara_prom
#
#
#  @param Matriz1 matriz con numeros
#  @param Matrizn Hasta N numeros de matriz de entradas a probar
#
#  @return true
##  
    def test_cara_promedio(self):
        matriz1 = [[1, 2, 3], [0, 0, 0],
                    [-0.23, 0.3, -9]]
        matriz2 = [[10, 223, 43], [10, 10, 10],
                    [-0.23234, 0.35, -69]]
        matriz3 = [[0, 0, 0], [0, 0, 0],
                    [0, 0, 0]]
        vectores = []
        vectores += [Trozos.convertirMatrizAVector(matriz1)]
        vectores += [Trozos.convertirMatrizAVector(matriz2)]
        vectores += [Trozos.convertirMatrizAVector(matriz3)]
        
        matriz = Trozos.crearMatrizDeVectores(vectores)
        cara_prom = (np.array(vectores[0]) +
                          np.array(vectores[1]) + np.array(vectores[2]))/3
        try:
            funcion = (Trozos.cara_prom(np.array(matriz)))
        except:
            self.assertEqual(True, False)
##
#  Prueba de integración entre las funciones cara_prom
#  y matriz_diferencias
#
#
#  @param Matriz1 matriz con numeros
#  @param Matrizn Hasta N numeros de matriz de entradas a probar
#
#  @return true
##  
    
    def test_matriz_diferencias_cara_promedio(self):
        matriz1 = [[1, 2, 3], [0, 0, 0],
                    [-0.23, 0.3, -9]]
        matriz2 = [[10, 223, 43], [10, 10, 10],
                    [-0.23234, 0.35, -69]]
        matriz3 = [[0, 0, 0], [0, 0, 0],
                    [0, 0, 0]]
        vectores = []
        vectores += [Trozos.convertirMatrizAVector(matriz1)]
        vectores += [Trozos.convertirMatrizAVector(matriz2)]
        vectores += [Trozos.convertirMatrizAVector(matriz3)]
        
        matriz = Trozos.crearMatrizDeVectores(vectores)
        matriz = np.array(matriz)
        cara_promedio = Trozos.cara_prom(np.array(matriz))
                          
        try:
            Trozos.matriz_diferencias(matriz, cara_promedio)
        except:
            self.assertEqual(True, False) 
##
#  Prueba de integración entre las funciones auto_vectores
#  y matriz_diferencias
#
#
#  @param Matriz1 matriz con numeros
#  @param Matrizn Hasta N numeros de matriz de entradas a probar
#
#  @return true
##  
    def test_autoVectores(self):
        matriz1 = [[1, 2, 3], [0, 0, 0],
                    [-0.23, 0.3, -9]]
        matriz2 = [[10, 223, 43], [10, 10, 10],
                    [-0.23234, 0.35, -69]]
        matriz3 = [[0, 0, 0], [0, 0, 0],
                    [0, 0, 0]]
        vectores = []
        vectores += [Trozos.convertirMatrizAVector(matriz1)]
        vectores += [Trozos.convertirMatrizAVector(matriz2)]
        vectores += [Trozos.convertirMatrizAVector(matriz3)]
        
        matriz = Trozos.crearMatrizDeVectores(vectores)
        matriz = np.array(matriz)
        cara_promedio = Trozos.cara_prom(np.array(matriz))
                        
        diferencias = Trozos.matriz_diferencias(matriz, cara_promedio)
        try:
            eigen = Trozos.auto_vectores(diferencias, 100)
        except:
            self.assertEqual(True, False) 
    
if __name__ == "__main__":
    unittest.main()
