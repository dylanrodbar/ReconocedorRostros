import unittest
from Clases import Entrenamiento
import numpy as np

##
#  Clase compuesta por las pruebas unitarias del archivo Clases.py
#
#
#  @param  TestCase Caso de uso a probar
#
##


class TestMyModule(unittest.TestCase):

    ##
    #  Prueba unitaria para la función convertirMatrizAVector.
    #  Prueba tres casos de uso:
    #       1. Una matriz de 2x3
    #       2. Una matriz vacia
    #       3. Una vector con cero
    #  @param
    #
    #  @return
    ##

    def test_convertirMatrizAVector(self):
        entrenamiento = Entrenamiento()
        self.assertEqual(entrenamiento.convertirMatrizAVector(
            [[1, 2, 3], [4, 5, 6]]), [1, 2, 3, 4, 5, 6])
        self.assertEqual(entrenamiento.convertirMatrizAVector([]), [])
        self.assertEqual(entrenamiento.convertirMatrizAVector([0]), [0])

    ##
    #  Prueba unitaria para la función crearMatrizDeVectores.
    #  Prueba tres casos de uso:
    #       1. Una matriz de 2x4
    #       2. Una matriz vacia
    #       3. Una matriz 1x1 con 1
    #
    #  @param
    #
    #  @return
    ##

    def test_crearMatrizDeVectores(self):
        entrenamiento = Entrenamiento()
        self.assertEqual(entrenamiento.crearMatrizDeVectores(
            [[1, 2, 3, 4], [5, 6, 7, 8]]), [[1, 5], [2, 6], [3, 7], [4, 8]])
        self.assertEqual(entrenamiento.crearMatrizDeVectores([]), [])
        self.assertEqual(entrenamiento.crearMatrizDeVectores([[1]]), [[1]])

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
        entrenamiento = Entrenamiento()
        self.assertEqual(entrenamiento.cara_prom(
            np.array([[11, 10, 4], [100, 200, 20]])).tolist(), [25/3, 320/3])
        self.assertEqual(entrenamiento.cara_prom(
            np.array([[0, 0, 0, 0], [0, 0, 0, 0]])).tolist(), [0, 0])

    ##
    #  Prueba unitaria para la funcion matriz_diferencias.
    #  Prueba el siguiente caso:
    #       1. Entradas: Matriz 2x4, Arreglo 1x4
    #       2. Salida: Arreglo 1x4 de ceros
    #
    #  @param
    #
    #  @return
    ##

    def test_matriz_diferencias(self):
        entrenamiento = Entrenamiento()
        self.assertEqual(entrenamiento.matriz_diferencias(
            np.array([[1, 2, 3, 4], [1, 2, 3, 4]], [1, 2, 3, 4])).tolist(),
                          [[0, 0, 0, 0], [0, 0, 0, 0]])

    ##
    #  Prueba unitaria para la funcion normalizar.
    #  Prueba el siguiente caso:
    #       1. Entradas: Arreglo 1x5
    #       2. Salida: 1
    #
    #  @param
    #
    #  @return
    ##

    def test_normalizar(self):
        entrenamiento = Entrenamiento()
        self.assertAlmostEqual(np.linalg.norm(entrenamiento.normalizar(
            [123.0032, 123.4, 36.0043, 3, 0, 345.19])), 1, None, 0.1)

    ##
    #  Prueba unitaria para la funcion normalizar.
    #  Prueba el siguiente caso:
    #       1. Entradas: Matriz 3x3
    #       2. Salida: Autovectores de la matriz con norma 1
    #
    #  @param
    #
    #  @return
    ##

    def test_auto_vectores(self):
        entrenamiento = Entrenamiento()
        covarianzaV = np.matmul(np.transpose(
            np.array([[1, 2, 3], [3, 4, 5], [5, 7, 8]])), np.array(
                [[1, 2, 3], [3, 4, 5], [5, 7, 8]]))
        eigen = np.linalg.eig(covarianzaV)
        auto_vectores = np.matmul(
            np.array([[1, 2, 3], [3, 4, 5], [5, 7, 8]]), eigen[1])
        autovecnorm = []
        for i in range(0, len(auto_vectores[0])):
            autovecnorm += [entrenamiento.normalizar(auto_vectores[:, i])]

        self.assertEqual(entrenamiento.auto_vectores(
            np.array([[1, 2, 3], [3, 4, 5], [5, 7, 8]])), autovecnorm)
