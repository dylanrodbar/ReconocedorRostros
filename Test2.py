import unittest
from Clases import *
import numpy as np

class TestMyModule(unittest.TestCase):
    
    def test_convertirMatrizAVector(self):
        entrenamiento = Entrenamiento()
        self.assertEqual(entrenamiento.convertirMatrizAVector([[1,2,3],[4,5,6]]), [1,2,3,4,5,6])
        self.assertEqual(entrenamiento.convertirMatrizAVector([]), [])
        self.assertEqual(entrenamiento.convertirMatrizAVector([0]), [0])
    
    def test_crearMatrizDeVectores(self):
        entrenamiento = Entrenamiento()
        self.assertEqual(entrenamiento.crearMatrizDeVectores([[1,2,3,4],[5,6,7,8]]), [[1,5],[2,6],[3,7],[4,8]])
        self.assertEqual(entrenamiento.crearMatrizDeVectores([]), [])
        self.assertEqual(entrenamiento.crearMatrizDeVectores([[1]]), [[1]])
        
    def test_cara_prom(self):
        entrenamiento = Entrenamiento()
        self.assertEqual(entrenamiento.cara_prom(np.array([[11,10,4],[100,200,20]])).tolist(), [25/3,320/3])
        self.assertEqual(entrenamiento.cara_prom(np.array([[0,0,0,0],[0,0,0,0]])).tolist(), [0,0])
        
    def test_matriz_diferencias(self):
        entrenamiento = Entrenamiento()
        self.assertEqual(entrenamiento.matriz_diferencias(np.array([[1,2,3,4],[1,2,3,4]],[1,2,3,4])).tolist(), [[0,0,0,0],[0,0,0,0]])
    
    def test_normalizar(self):
        entrenamiento = Entrenamiento()
        self.assertAlmostEqual(np.linalg.norm(entrenamiento.normalizar([123.0032, 123.4,36.0043,3,0,345.19])), 1,None,0.1 )
        
    def test_auto_vectores(self):
        entrenamiento = Entrenamiento()
        covarianzaV = np.matmul(np.transpose(np.array([[1,2,3],[3,4,5],[5,7,8]])),
                                np.array([[1,2,3],[3,4,5],[5,7,8]]))
        eigen = np.linalg.eig(covarianzaV)
        auto_vectores = np.matmul(np.array([[1,2,3],[3,4,5],[5,7,8]]), eigen[1])
        autovecnorm = []
        for i in range(0, len(auto_vectores[0])):
            autovecnorm += [entrenamiento.normalizar(auto_vectores[:, i])]
        
        self.assertEqual(entrenamiento.auto_vectores(np.array([[1,2,3],[3,4,5],[5,7,8]])), autovecnorm )