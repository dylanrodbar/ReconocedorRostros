import unittest
import Trozos
import numpy as np

class TestMyModule(unittest.TestCase):
    
    
    def test_cargarImagen(self):
        self.assertEqual(Trozos.cargarImagen('input\s1'), None,'No se pudo cargar la imagen')
     
    def test_convertirMatrizAVector(self):
        self.assertEqual(Trozos.convertirMatrizAVector([[1,2,3],[4,5,6]]), [1,2,3,4,5,6])
        self.assertEqual(Trozos.convertirMatrizAVector([]), [])
        self.assertEqual(Trozos.convertirMatrizAVector([0]), [0])
        
    def test_crearMatrizDeVectores(self):
        self.assertEqual(Trozos.crearMatrizDeVectores([[1,2,3,4],[5,6,7,8]]), [[1,5],[2,6],[3,7],[4,8]])
        self.assertEqual(Trozos.crearMatrizDeVectores([]), [])
        self.assertEqual(Trozos.crearMatrizDeVectores([[1]]), [[1]])
        
    def test_calcularMatrizCovarianza(self): 
        self.assertEqual(Trozos.calcularMatrizCovarianza([[1,2,3,4]]), np.array(1.6666666666666665))  
    
if __name__ == "__main__":
    unittest.main()
