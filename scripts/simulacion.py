# _____________________________________________________________________________
#
# Proyecto:       Simulacion_Gillingham
#
# Script:         scripts/simulacion.py
# Objetivo:
#
# Autor:          Rodrigo Antonio Aldrette Salas
# Correo(s):      raaldrettes@colmex.mx
#
# Fecha:          29/01/2026
# 
# Última
# actualización:  29/01/2026
#
# _____________________________________________________________________________

# PREAMBULO ___________________________________________________________________

import pandas as pd
import numpy as np


# CODIGO ______________________________________________________________________

# Paso 1: Establecer parámetros ---------------------------------------------------------------

# Factor de descuento
beta = 0.95

# Hay dos tipos de hogares: ricos y pobres


# Hay 3 coches con distintas calidades y precios
x = [10, 20, 30] # Utilidad de los coches
p_n = [100, 170, 250] # Precios de nuevos (promedio)
# Su scrap value
p_s = [1, 5, 8]

# Los costos de transacción
t_b = 10
t_s = 10

# Depreciación por edad
delta = 1

# Funciones de utilidad
def util(x_j, edad, gasto, rico=0):
    global delta
    # Mu es su sensibilidad a la pérdida de dinero (efecto riqueza)
    mus = [20, 10]
    mu = mus[rico]
    valor = x_j - delta * edad - mu * (gasto)
    return valor



# Paso 2: Encontrar precios P ---------------------------------------------------------------

# Paso 3: Simulación ---------------------------------------------------------------
