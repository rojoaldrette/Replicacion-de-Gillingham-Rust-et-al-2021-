# _____________________________________________________________________________
#
# Proyecto:       Simulacion_Gillingham
#
# Script:         scripts/estimacion.py
# Objetivo:
#
# Autor:          Rodrigo Antonio Aldrette Salas
# Correo(s):      raaldrettes@colmex.mx
#
# Fecha:          08/01/2026
# 
# Última
# actualización:  29/01/2026
#
# _____________________________________________________________________________


# PREAMBULO ___________________________________________________________________

import pandas as pd
import numpy as pd


# Ahora intentaremos obtener los parámetros que definimos en config a través
# de los métodos del artículo. La ventaja del planteamiento del artículo es la ausen-
# cia de necesidad de usar un método de variables instrumentales.
# Este artículo (Gillingham, 2021) lo comparo directamente con Schiraldi(2011)
# el cual hace un modelo muy parecido, pero sin un equilibrio estacionario, sino
# que los consumidores tienen un conjunto de información y expectativas del futuro
# sobre el cual forman su Bellman. Naturalmente ese approach es más sensible a la
# convergencia.

# Pasos a seguir:
# 1. 