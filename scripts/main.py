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
    print("Calculando valor crítico")
    valor_critico = simulacion.ejecutar_optimizacion()
    print("Valor crítico calculado")
    
    # 2. Pasar ese valor a datos.py para procesarlo
    print("Generando datos...")
    df_owner, df_trans, df_owner_trans = generar_datos.gen_datos(valor_critico, 10000, 7)
    df_owner.to_csv('output/df_owner.csv')
    df_trans.to_csv('output/df_trans.csv')
    df_owner_trans.to_csv('output/df_owner_trans.csv')
    print("Los datos se han generado!")


if __name__ == "__main__":
    run()

