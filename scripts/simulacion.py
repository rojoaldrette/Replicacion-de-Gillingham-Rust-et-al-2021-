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
# Edad máxima
A_max = 15
# Marcas
J = 3
# Tipos de agentes (rico y pobre), 1 = pobre, 2 = rico
n_types = 2
# Sus mu's (efectos riqueza)
mus = np.array([20, 10])
# Correlación entre elecciones
sigma = 1



# Hay 3 coches con distintas calidades y precios
calidad_x = np.array([10, 20, 30]) 
precios_nuevos = np.array([100, 170, 250])
scrap_values = np.array([1, 5, 8])

# Los costos de transacción
t_b = 10
t_s = 10

# Depreciación por edad
delta = 1


# Paso 2. Infraestructura y utilidad ---------------------------------------------------------

# -- Espacio de Estados ----------
# Estado 0: No tener coche
# Estado 1: 1,...,A_max (tener coche 1)
# Estado 2: A_max+1, ..., 2*A_max (tener coche 2)
n_states = 1 + J * A_max

# Índice lineal para los estados
def get_idx(j, a):
    if j is None: return 0 # Estado "sin coche"
    return 1 + j * A_max + (a - 1)


# --- Utilidad de cada estado -----
# Creamos una matriz, u_vector, con la utilidad de cada estado (tipo, estado).
# Calculamos la utilidad base de existir en cada estado (tenencia)

u_matrix = np.zeros((n_types, n_states))

for tau in range(n_types):
    # Utilidad de no tener coche
    u_matrix[tau, 0] = 0 
    
    for j in range(J):
        for a in range(1, A_max + 1):
            idx = get_idx(j, a)
            # Calculamos la utilidad a partir de lo que da de calidad menos la depreciación
            u_matrix[tau, idx] = calidad_x[j] - delta * a



# --- Matriz de envejecimiento y deshuesadero -----------
# Creamos Q_a como la matriz de transicion de una edad a otra
# P.D. QUE INCREIBLE MANEJAR MATRICES DE TRANSICIÓN EN PROGRAMACIÓN
# ES TAN INTUITIVO: Q_a[desde, hacia] = probabilidad

# Probabilidad de quedar en edad terminal según su modelo y edad
def alpha(j, a):
    if a == (A_max - 1):
        prob = 1
    else:
        # Función creciente en a y decreciente en j (calidad)
        prob = (a/(4 * A_max)) + ((3-j) / 20)
    return prob


Q_a = np.zeros((n_states, n_states))
# Los del estado cero tienen p=1 de seguir sin coche
Q_a[0, 0] = 1
# Lista de edades máximas de coches
max_ages = [get_idx(j, A_max) for j in range(J)]

for j in range(J):
    for a in range(1, A_max + 1):
        # Índice inicial
        idx_0 = get_idx(j, a)
        if idx_0 not in max_ages:
            # Índice destino
            idx_1 = get_idx(j, a + 1)
            # Prob de sobrevivir y pasar a la siguiente edad
            Q_a[idx_0, idx_1] = 1 - alpha(j, a)
            # Prob de falla
            Q_a[idx_0, max_ages[j]] = alpha(j, a)
    
    # Garantizamos que si estas en edad máxima -> desguezadero
    idx_max = get_idx(j, A_max)
    Q_a[idx_max, 0] = 1

# Verificar si suman a 1 las filas (por construcción no debería haber problema),
# me sirvió de debugging
if not np.allclose(Q_a.sum(axis=1), 1):
    print("Error: Hay filas en Q_a que no suman 1. Revisa los índices.")

# Paso 3: Bellman ----------------------------------------------------------------------

# Función para crear vector de precios para cada estado
def get_P(precios_usados):
    # Creamos vector de precios
    P = np.zeros(n_states)
    # Precio de no coche es cero
    P[0] = 0

    k = 0
    for j in range(J):
        for a in range(1, A_max + 1):
            idx = get_idx(j, a)
            if a == 1:
                P[idx] = precios_nuevos[j]
            elif a == A_max:
                P[idx] = scrap_values[j]
            else:
                P[idx] = precios_usados[k]
                k += 1
    return P


def sol_bellman(precios_usados):
    """
    Esta función encuentra el punto fijo EV para la bellman con los parámetros que creamos.
    Toma un vector de precios de coches usados (excluidos los exógenos: nuevos y scrap), lo
    acomoda en un vector de precios y después crea la matriz EV (tipos x estados) a partir
    de la utilidad máxima que puede conseguir.

    Hay un segundo objetivo en esta función: crear la matriz u_trade, la cual nos da la utilidad
    inmediata de cada decisión según el vector P. Esta nos facilitará la creación de las probabilidades.
    Puede que no sea lo más eficiente (tener que crearlo cada vez que se corre esta función). Pero
    nos permite matar dos pajaros de un tiro en términos de redundancia del código.
    """

    P = get_P(precios_usados)

    # Hacemos la matriz de funciones de valor
    # Inicializamos en ceros
    EV = np.zeros((n_types, n_states))
    EV_new = np.zeros_like(EV)

    # Iteramos para encontrar punto fijo (decidí no hacer un algorítmo Newton por el tiempo)
    # Aquí los consumidores de distintos estados se encontrarán con sus posibles elecciones
    for _ in range(500):
        # Valor inicial de E[V]=EV*Q_a.T
        EV_next = np.dot(EV, Q_a.T)

        for tau in range(n_types):
            mu = mus[tau]

            for s0 in range(n_states):
                # Lista de opciones
                v_alternativas = []
                # Valor de venta del coche
                val_venta = 0

                # MANTENER --------------------------------
                ## Sin coche
                if s0 == 0:
                    v_keep = u_matrix[tau, 0] + beta * EV_next[tau, 0]
                    v_alternativas.append(v_keep)
                ## Con coche
                else:
                    v_keep = u_matrix[tau, s0] + beta * EV_next[tau, s0]
                    v_alternativas.append(v_keep)

                    # Ponemos el valor de venta
                    val_venta = (P[s0] - t_s) * mu
                
                # COMPRAR COCHE -------------------------
                # Iteramos sobre todos los coches (incluido el no coche)
                for s1 in range(n_states):
                    # Si es un estado sin coche
                    costo_compra = (P[s1] + t_b) * mu
                    if s1 == 0:
                        costo_compra = 0
                    # Si el coche está en edad terminal
                    elif s1 in max_ages:
                        # Si el coche tiene edad máxima, entonces no se puede comprar
                        # Se puede comprar máximo en edad A_max - 1
                        # Continuamos a la siguiente iteración
                        continue
                    # Obtenemos utilidad de adquirir cada coche
                    u_trade = u_matrix[tau, s1] - costo_compra + val_venta + beta * EV_next[tau, s1]
                    v_alternativas.append(u_trade)

                # Calculamos su EV a partir del valor máximo que puede conseguir
                vals = np.array(v_alternativas)
                max_val = np.max(vals)
                # Hacemos esta transformación para no explotar exp(vals) -> inf (Gracias ChatGPT)
                ev_val = max_val + sigma * np.log(np.sum(np.exp((vals - max_val)/ sigma)))
                EV_new[tau, s0] = ev_val

        # Si es menor al threshold, entonces decimos que es el punto fijo
        if np.max(np.abs(EV_new - EV)) < 1e-16:
            break
        # Copy porque namas así controlas los pointers del python xd (En R no es necesario, 
        # pero siento que arriesgas más memory overflow con estos procesos)
        EV = EV_new.copy()
    
    # Regresamos el punto fijo
    return EV



# Paso 4: Demanda y oferta ---------------------------------------------------------------

# ------- Probabilidades ----------------------

def calc_probs(EV, p_guess):
    """
    Para calcular las probabilidades vamos a basarnos en la fórmula:
    Pi[s1|s0] = exp((v_{s1}(s0) - EV(s)) / sigma)
    que es idéntica a la del paper, pero no explota en python.
    Hay que llenar la matriz omega[s0, s1] = Pi[s1 | s0].
    Aquí v_{s1}(s0) es el valor de u_matrix[,s0] + trade_costs
    """
    return 0


def ED(P, omega):
    return 0


# Paso 3: Simulación ---------------------------------------------------------------



