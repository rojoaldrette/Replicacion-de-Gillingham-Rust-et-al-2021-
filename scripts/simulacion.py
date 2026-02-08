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

# Notas para arreglar:
# Chance corregir alpha y su intima relación con el coche de calidad 1
# No se están vendiendo coches de calidad 1 y 2

# PREAMBULO ___________________________________________________________________

import pandas as pd
import numpy as np
import time
from datetime import timedelta
from scipy.special import logsumexp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

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
# Proporción agentes
type_mass = [0.5, 0.5]
# Sus mu's (efectos riqueza)
mus = np.array([0.5, 0.2])
# Correlación entre elecciones
sigma = 2



# Hay 3 coches con distintas calidades y precios
calidad_x = np.array([17, 50, 50]) 
precios_nuevos = np.array([80, 140, 350])
scrap_values = np.array([7, 8, 10])

# Los costos de transacción
t_b = 0
t_s = 0

# Depreciación por edad
delta = 0.8


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
# Inversos a get_idx
def get_brand(state):
    if state == 0:
        return None
    return (state - 1) // A_max
def get_age(state):
    if state == 0:
        return 0
    return ((state - 1) % A_max) + 1

# Creamos un vector de scrap values para cada situación
scrap_vec = np.zeros(n_states)
for s in range(n_states):
    if s == 0:
        scrap_vec[s] = 0
    else:
        marca = get_brand(s) # Debería devolver 0, 1 o 2 (a veces float xd)
        scrap_vec[s] = scrap_values[int(marca)]

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
        prob = (a/(5 * A_max)) + ((3-j) / 20)
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
    P = np.zeros(n_states)
    # 1. Ponemos precios de nuevos y scrap (fijos)
    for j in range(J):
        P[get_idx(j, 1)] = precios_nuevos[j]
        P[get_idx(j, A_max)] = scrap_values[j]
    
    idx_usados = get_indices_usados()
    P[idx_usados] = precios_usados
    return P



# Paso 4: Probabilidades  ---------------------------------------------------------------


def calc_probs(EV, precios_guess, tau):
    """
    Esta función nos debe dar las probabilidades necesarias para calcular omega en su totalidad y
    las probabilidades de chatarrización endógena para calcular el exceso de demanda
    """
    P = get_P(precios_guess)
    mu = mus[tau]
    EV_next = np.dot(EV, Q_a.T)

    # ------ Prob scrap y sell --------------------------------------
    # Obtener probabilidades de scrap y sell para incluirlo en las probabilidades de omega
    u_vender = mu * (P - t_s)
    u_scrap = mu * scrap_vec
    for age in max_ages:
        u_vender[age] = -np.inf
    
    m_salida = np.maximum(u_vender, u_scrap)
    v_disposal = m_salida + sigma * np.log(np.exp((u_vender - m_salida)/sigma) + 
                                           np.exp((u_scrap - m_salida)/sigma))
    # Probabilidad condicional de elegir SCRAP dado que se hace TRADE
    # prob_scrap_cond[s0]
    p_scrap_cond = np.exp((u_scrap - v_disposal)/sigma)
    
    # --- Funciones de valor --------------------------

    ## MANTENER ----------------------------
    v_keep = u_matrix[tau, :] + beta * EV_next[tau, :]
    # Hacer mantener imposible si está en edad terminal
    for age_max_idx in max_ages:
        v_keep[age_max_idx] = -np.inf
    
    ## TRADE ---------------------------------
    # Matriz de transacciones: v_trade[s0, s1]
    # Usamos broadcasting de numpy para evitar los loops de s0 y s1
    # Utilidad de tener s1 - costo compra s1 + valor venta s0 + valor futuro s1
    v_trade = (u_matrix[tau, :][None, :] 
               - mu * (P + t_b)[None, :] 
               + v_disposal[:, None] # El valor esperado de su salida
               + beta * EV_next[tau, :][None, :])
    
    # No se pueden comprar coches en edad terminal
    for age_max_idx in max_ages:
        v_trade[:, age_max_idx] = -np.inf

    # Quitamos el costo de búsqueda para ir al s=0
    v_trade[:, 0] = u_matrix[tau, 0] + v_disposal + beta * EV_next[tau, 0]

    # ---------- MATRIZ DE TRANSICIÓN OMEGA -------------------------------------------
    
    # Para calcular omega[s0, s1], necesitamos la probabilidad de CADA opción.
    # Pero omega solo registra el estado final s1. 
    # Si s1 == s0, hay dos formas de estar ahí: haber mantenido (keep) 
    # o haber vendido y comprado el mismo modelo (trade).
    
    # Usamos el EV[tau, s0] que ya calculamos en sol_bellman como denominador
    # Esto garantiza estabilidad numérica total.
    
    prob_keep = np.exp((v_keep - EV[tau, :]) / sigma)
    prob_trade_matrix = np.exp((v_trade - EV[tau, :][:, None]) / sigma)
    
    # La matriz Omega final:
    omega_tau = prob_trade_matrix.copy()
    # Sumamos la probabilidad de mantener en la diagonal (o donde s1 == s0)
    np.fill_diagonal(omega_tau, np.diag(omega_tau) + prob_keep)

    if np.any(np.isnan(omega_tau)) or np.any(np.isinf(omega_tau)):
        raise FloatingPointError("NaN o Inf detectado en omega_tau")

    # Retornamos omega para calcular q y p_scrap_cond para calcular el exceso de demanda
    return omega_tau, p_scrap_cond, prob_keep



# Paso 5: Población  ---------------------------------------------------------------


def get_q_tau(omega):

    # Matriz de transición completa
    T = omega @ Q_a

    n = T.shape[0]
    q = np.ones(n) / n
    for _ in range(5000):
        q_new = q @ T
        if np.max(np.abs(q_new - q)) < 1e-13:
            return q_new
        q = q_new
    return q

def get_big_q(q_list):
    # Calculamos la q final a partir de las q's que le damos en lista
    big_q = np.zeros(n_states)
    for i, q in enumerate(q_list):
        adj_q = type_mass[i] * q
        big_q = big_q + adj_q
    return big_q


# Paso 6: Oferta y demanda ---------------------------------------------------------


def get_indices_usados():
    """
    Identifica los índices de los estados que corresponden a coches usados.
    Estos son los mercados cuyos precios el optimizador debe ajustar.
    """
    indices_usados = []
    
    for s in range(n_states):
        # 1. Ignoramos el estado s=0 (no hay bien que vender)
        if s == 0:
            continue
            
        # 2. Obtenemos la edad del coche en este estado
        age = get_age(s)
        
        # 3. Filtramos:
        # - Debe ser mayor a 0 (no es nuevo)
        # - No debe estar en la lista de edades terminales (no se vende a particulares)
        if age > 1 and s not in max_ages:
            indices_usados.append(s)
            
    return np.array(indices_usados)

    
def sol_bellman_vectorized(precios_usados):
    P = get_P(precios_usados)
    EV = np.zeros((n_types, n_states))
    
    # Pre-calculamos utilidades de salida (v_disposal) para todos los estados
    # s0=0 -> 0, s0>0 -> logsumexp(vender, chatarrizar)
    u_vender_base = (P - t_s) # Vector de n_states
    u_scrap_base = scrap_vec
    
    for _ in range(500):
        EV_next = EV @ Q_a.T
        EV_old = EV.copy()
        
        for tau in range(n_types):
            mu = mus[tau]
            
            # Valor de deshacerse del coche actual
            u_v = mu * u_vender_base
            u_v[max_ages] = -np.inf # No se venden terminales
            u_v[0] = -np.inf        # No se vende el "no tener coche"
            u_s = mu * u_scrap_base
            u_s[0] = -np.inf
            
            # v_disposal[s0]
            v_disposal = sigma * logsumexp(np.vstack([u_v, u_s]) / sigma, axis=0)
            v_disposal[0] = 0 # El que no tiene coche no gana nada por "vender"
            
            # Opción MANTENER: v_keep[s0]
            v_keep = u_matrix[tau, :] + beta * EV_next[tau, :]
            v_keep[max_ages] = -np.inf # Forzado a chatarrizar
            
            # Opción TRADE: v_trade[s0, s1] 
            # (Utilidad de s1 - costo s1 + v_disposal de s0 + valor futuro s1)
            # Usamos broadcasting: (n_states, 1) + (1, n_states)
            v_trade = (u_matrix[tau, :][None, :] 
                       - mu * (P + t_b)[None, :] 
                       + v_disposal[:, None] 
                       + beta * EV_next[tau, :][None, :])
            
            v_trade[:, max_ages] = -np.inf # No se compran terminales
            # Opción especial: Volver a s=0 (No comprar nada tras vender)
            v_trade[:, 0] = 0 + v_disposal + beta * EV_next[tau, 0]

            # Bellman Update: LogSumExp sobre todas las opciones
            # Para cada s0, el agente elige entre {mantener, trade_s1, trade_s2...}
            choices = np.hstack([v_keep[:, None], v_trade]) # (n_states, 1 + n_states)
            EV[tau, :] = sigma * logsumexp(choices / sigma, axis=1)

        if np.max(np.abs(EV - EV_old)) < 1e-8:
            break
    return EV

def ED(p_guess):
    EV = sol_bellman_vectorized(p_guess)
    P = get_P(p_guess)
    
    total_demand = np.zeros(n_states)
    total_supply = np.zeros(n_states)
    
    for tau in range(n_types):
        mu = mus[tau]
        EV_next = EV @ Q_a.T
        
        # 1. Probabilidades de salida (Scrap vs Sell)
        u_v = mu * (P - t_s)
        u_v[max_ages] = -1e10
        u_s = mu * scrap_vec
        # Prob condicional de scrap dado que decides salir
        # P(scrap | disposal) = exp(u_s/sigma) / (exp(u_s/sigma) + exp(u_v/sigma))
        p_scrap_cond = np.exp(u_s/sigma) / (np.exp(u_s/sigma) + np.exp(u_v/sigma))
        p_scrap_cond[0] = 0

        # 2. Matrices de decisión
        v_keep = u_matrix[tau, :] + beta * EV_next[tau, :]
        v_keep[max_ages] = -1e10
        
        # v_disposal igual que en Bellman
        v_disposal = sigma * logsumexp(np.vstack([mu*(P-t_s), mu*scrap_vec])/sigma, axis=0)
        v_disposal[0] = 0
        v_disposal[max_ages] = mu * scrap_vec[max_ages] # Forzado a scrap
        
        v_trade = (u_matrix[tau, :][None, :] - mu*(P+t_b)[None, :] + 
                   v_disposal[:, None] + beta*EV_next[tau, :][None, :])
        v_trade[:, max_ages] = -1e10
        v_trade[:, 0] = v_disposal + beta*EV_next[tau, 0]

        # Probabilidades Logit puras (sin normalizar a mano, el denominador es exp(EV/sigma))
        prob_keep = np.exp((v_keep - EV[tau, :]) / sigma)
        prob_trade_mat = np.exp((v_trade - EV[tau, :][:, None]) / sigma)
        
        # 3. Distribución Estacionaria
        omega = prob_trade_mat.copy()
        np.fill_diagonal(omega, np.diag(omega) + prob_keep)
        q_t = get_q_tau(omega)
        
        # 4. Oferta y Demanda
        # Demanda de s1: sum_{s0} q(s0) * P(trade to s1 | s0)
        demand_tau = q_t @ prob_trade_mat
        
        # Oferta de s0: q(s0) * P(no keep) * P(no scrap | no keep)
        supply_tau = q_t * (1 - prob_keep) * (1 - p_scrap_cond)
        
        total_demand += type_mass[tau] * demand_tau
        total_supply += type_mass[tau] * supply_tau

    idx_u = get_indices_usados()
    return (total_demand - total_supply)[idx_u]


# Paso 7: Encontrar P -------------------------------------------------------------

idx_usados = get_indices_usados()
p_init = np.zeros(len(idx_usados))

# Gemini me recomendó meterle un guess lineal de caída de precios
# En general, por mi desconocimiento de scipy, gemini me ayudó mucho en cerrar esta parte
for i, s_idx in enumerate(idx_usados):
    p_new = precios_nuevos[get_brand(s_idx)] # Precio del coche nuevo de esa marca
    edad = get_age(s_idx)
    # Suponemos que cada año el valor cae un 15% como punto de partida
    p_init[i] = p_new * (0.85 ** edad)


def ED_wrapper(p_vec):
    error_vec = ED(p_vec)
    if np.any(np.isnan(error_vec)) or np.any(np.isinf(error_vec)):
        print("ED tiene NaN/Inf. p_vec mínimo/máximo:", np.min(p_vec), np.max(p_vec))
        raise FloatingPointError("ED contiene NaN/Inf")
    norma_error = np.linalg.norm(error_vec)
    print(f"Iteración: Error promedio (norma): {norma_error:.8f}")
    return error_vec

# -------------------------------------------------

print("Iniciando la búsqueda del equilibrio de mercado...")



# Scipy shit ---------------------------------------
eps = 1e-6
lb = np.ones_like(p_init) * eps
ub = np.ones_like(p_init) * 1e6


start_time = time.perf_counter()

res = least_squares(
    fun=ED_wrapper,
    x0=p_init,
    bounds=(1e-6, np.inf),
    xtol=1e-8,
    ftol=1e-8,
    gtol=1e-8,
    max_nfev=1000
)
if res.success:
    precios_equilibrio_usados = res.x
    print("least_squares convergió. norm error:", np.linalg.norm(ED(precios_equilibrio_usados)))
else:
    print("least_squares falló:", res.message)

end_time = time.perf_counter()
elapsed_time_secs = end_time - start_time

# For a human-readable format
msg = f"Execution took: {timedelta(seconds=round(elapsed_time_secs))} (Wall clock time)"
print(msg)


# Graficas
def graficar_distribucion(p_final):
    # 1. Obtenemos el EV final y las distribuciones
    EV = sol_bellman_vectorized(p_final)
    dist_tipos = []
    
    for tau in range(n_types):
        omega_t, _, p_keep_t = calc_probs(EV, p_final, tau) # Usa tu calc_probs corregida
        q_t = get_q_tau(omega_t)
        dist_tipos.append(q_t)

    # 2. Preparar datos para graficar por marca
    fig, axes = plt.subplots(1, J, figsize=(15, 5), sharey=True)
    nombres_tipos = ['Pobre (mu=0.3)', 'Rico (mu=0.1)']
    colores = ['#e74c3c', '#3498db']

    for j in range(J):
        ax = axes[j]
        edades = np.arange(1, A_max + 1)
        
        for tau in range(n_types):
            # Extraemos la masa de cada edad para la marca j
            indices_marca = [get_idx(j, a) for a in edades]
            masa_marca = dist_tipos[tau][indices_marca]
            
            ax.plot(edades, masa_marca, label=nombres_tipos[tau], 
                    color=colores[tau], marker='o', markersize=4)
        
        ax.set_title(f"Marca {j+1} (Calidad: {calidad_x[j]})")
        ax.set_xlabel("Edad del coche")
        if j == 0: ax.set_ylabel("Densidad de población (q)")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.suptitle("Distribución Estacionaria de Tenencia de Vehículos", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 3. Print de estado "Sin Coche" (Estado 0)
    for tau in range(n_types):
        print(f"Tipo {tau+1} sin coche: {dist_tipos[tau][0]:.2%}")

graficar_distribucion(precios_equilibrio_usados)

# Paso 8: Generar datos con P ----------------------------------------------------



# Paso 3: Simulación ---------------------------------------------------------------



