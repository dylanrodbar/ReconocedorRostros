import unittest
import Trozos
import numpy as np
##
#  Clase compuesta por las pruebas unitarias del archivo Trozos.py 
#  
#
#  @param  TestCase Caso de uso a probar
#    
##
class TestMyModule(unittest.TestCase):
    
##
#  Prueba unitaria para la función cargarImagen.
#  Prueba como entrada una carpeta con imagenes.
#    
##
    def test_cargarImagen(self):
        self.assertEqual(Trozos.cargarImagen('input\s1'), None,'No se pudo cargar la imagen')
##
#  Prueba unitaria para la función convertirMatrizAVector.
#  Prueba tres casos de uso:
#       1. Una matriz de 2x3
#       2. Una matriz vacia
#       3. Una vector con cero
##     
    def test_convertirMatrizAVector(self):
        self.assertEqual(Trozos.convertirMatrizAVector([[1,2,3],[4,5,6]]), [1,2,3,4,5,6])
        self.assertEqual(Trozos.convertirMatrizAVector([]), [])
        self.assertEqual(Trozos.convertirMatrizAVector([0]), [0])
##
#  Prueba unitaria para la función crearMatrizDeVectores.
#  Prueba tres casos de uso:
#       1. Una matriz de 2x4
#       2. Una matriz vacia
#       3. Una matriz 1x1 con 1
#    
##        
    def test_crearMatrizDeVectores(self):
        self.assertEqual(Trozos.crearMatrizDeVectores([[1,2,3,4],[5,6,7,8]]), [[1,5],[2,6],[3,7],[4,8]])
        self.assertEqual(Trozos.crearMatrizDeVectores([]), [])
        self.assertEqual(Trozos.crearMatrizDeVectores([[1]]), [[1]])
##
#  Prueba unitaria para la función calcularMatrizCovarianza.
#  Prueba como entrada una matriz 1x4.
#    
##        
    def test_calcularMatrizCovarianza(self): 
        self.assertEqual(Trozos.calcularMatrizCovarianza([[1,2,3,4]]), np.array(1.6666666666666665))  
    
if __name__ == "__main__":
    unittest.main()
