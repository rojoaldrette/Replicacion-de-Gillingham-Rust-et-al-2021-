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


import numpy as np
import pandas as pd
from simulacion import sol_bellman_vectorized, calc_probs, get_idx, get_P, get_brand, get_age
import simulacion as sim


# CÓDIGO ______________________________________________________________________

def cargar_precios():
    try:
        precios = np.load("precios_optimizados.npy")
        return precios
    except FileNotFoundError:
        return "Error: No existe el archivo. Corre la simulación primero."

precios = get_P(cargar_precios())







if __name__ == "__main__":
    print("gen datos")
