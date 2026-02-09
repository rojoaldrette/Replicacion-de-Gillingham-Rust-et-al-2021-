# _____________________________________________________________________________
#
# Proyecto:       Simulacion_Gillingham
#
# Script:         scripts/generar_datos.py
# Objetivo:
#
# Autor:          Rodrigo Antonio Aldrette Salas
# Correo(s):      raaldrettes@colmex.mx
#
# Fecha:          08/02/2026
# 
# Última
# actualización:  29/01/2026
#
# _____________________________________________________________________________


# PREAMBULO ___________________________________________________________________


import numpy as np
import pandas as pd
import simulacion as sim



# CÓDIGO ______________________________________________________________________

def cargar_precios():
    try:
        precios = np.load("precios_optimizados.npy")
        return precios
    except FileNotFoundError:
        precios_optimos = sim.ejecutar_optimizacion()
        print("Error: No existe el archivo. Corriendo la simulación.")
        return precios_optimos

precios = cargar_precios()


def gen_datos(precios_usados, N, ):

    P = sim.get_P(precios_usados)
    q_final, q_ss, p_trade, omegas = sim.get_info(P)







if __name__ == "__main__":
    print("gen datos")
