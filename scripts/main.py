# _____________________________________________________________________________
#
# Proyecto:       Simulacion_Gillingham
#
# Script:         scripts/main.py
# Objetivo:
#
# Autor:          Rodrigo Antonio Aldrette Salas
# Correo(s):      raaldrettes@colmex.mx
#
# Fecha:          29/01/2026
# 
# Última
# actualización:  08/02/2026
#
# _____________________________________________________________________________


import simulacion
import generar_datos


def run():
    print("Iniciando proyecto...")
    # 1. Obtener el valor crítico de la simulación
    valor_critico = simulacion.ejecutar_optimizacion()
    
    # 2. Pasar ese valor a datos.py para procesarlo
    generar_datos.procesar_resultado(valor_critico)

if __name__ == "__main__":
    run()