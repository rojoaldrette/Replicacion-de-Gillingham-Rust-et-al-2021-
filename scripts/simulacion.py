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
from scipy.special import logsumexp
from scipy.optimize import root
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
mus = np.array([0.3, 0.1])
# Correlación entre elecciones
sigma = 1



# Hay 3 coches con distintas calidades y precios
calidad_x = np.array([10, 20, 30]) 
precios_nuevos = np.array([100, 170, 250])
scrap_values = np.array([1, 5, 8])

# Los costos de transacción
t_b = 5
t_s = 5

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


def sol_bellman(precios_usados):
    """
    Esta función encuentra el punto fijo EV para la bellman con los parámetros que creamos.
    Toma un vector de precios de coches usados (excluidos los exógenos: nuevos y scrap), lo
    acomoda en un vector de precios y después crea la matriz EV (tipos x estados) a partir
    de la utilidad máxima que puede conseguir.
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
                # Marca del coche

                # -- Valor de venta/scrap ------------------
                if s0 == 0:
                    v_disposal = 0
                else:
                    brand = get_brand(s0)
                    u_vender = mu * (P[s0] - t_s)
                    u_scrap  = mu * scrap_values[brand]
            
                    # Si el coche es terminal, no se puede vender a particular
                    if s0 in max_ages:
                        u_vender = -np.inf 
            
                    # Aplicamos Log-Sum-Exp para el valor de salida (v_disposal)
                    # Esto representa el valor esperado de elegir entre vender o chatarrizar
                    # i.e. sacar la probabilidad de escoger alguno de los dos
                    # Esto lo sacamos aquí para obtener las funciones de valor, pero
                    # en la siguiente función obtenemos su probabilidad para la demanda y oferta
                    m_salida = max(u_vender, u_scrap)
                    v_disposal = (sigma * logsumexp(np.array([u_vender, u_scrap]) / sigma))

                ### DECISIONES ---------------------------------------------------------------
                #-------- MANTENER --------------------------------
                ## Sin coche
                if s0 == 0:
                    v_keep = u_matrix[tau, 0] + beta * EV_next[tau, 0]
                    v_alternativas.append(v_keep)
                ## Con coche
                elif s0 in max_ages:
                    v_keep = -np.inf
                    v_alternativas.append(v_keep)
                else:
                    v_keep = u_matrix[tau, s0] + beta * EV_next[tau, s0]
                    v_alternativas.append(v_keep)
                
                # COMPRAR COCHE / TRADE -------------------------
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
                    u_trade = u_matrix[tau, s1] - costo_compra + v_disposal + beta * EV_next[tau, s1]
                    v_alternativas.append(u_trade)

                # Calculamos su EV a partir del valor máximo que puede conseguir
                vals = np.array(v_alternativas)
                max_val = np.max(vals)
                # Hacemos esta transformación para no explotar exp(vals) -> inf (Gracias ChatGPT)
                ev_val = sigma * logsumexp(vals / sigma)
                EV_new[tau, s0] = ev_val

        # Si es menor al threshold, entonces decimos que es el punto fijo
        if np.max(np.abs(EV_new - EV)) < 1e-9:
            break
        # Copy porque namas así controlas los pointers del python xd (En R no es necesario, 
        # pero siento que arriesgas más memory overflow con estos procesos)
        EV = EV_new.copy()

    # Regresamos el punto fijo
    return EV



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

    row_sums = omega_tau.sum(axis=1, keepdims=True)

    zero_rows = (row_sums.flatten() == 0)

    if np.any(zero_rows):
        # Si ocurre, asigna prob uniforme condicionalmente a alternativas válidas
        for i, zr in enumerate(zero_rows):
            if zr:
                valid = ~np.isneginf(v_trade[i, :])  # alternativas válidas
                if valid.sum() == 0:
                    omega_tau[i, 0] = 1.0  # fallback: quedarse sin coche
                else:
                    omega_tau[i, valid] = 1.0 / valid.sum()
        row_sums = omega_tau.sum(axis=1, keepdims=True)

    omega_tau = omega_tau / row_sums
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

def ED(p_guess):
    # 1. Resolver el problema del consumidor
    EV = sol_bellman(p_guess)
    
    # Inicializamos agregadores de mercado
    total_demand = np.zeros(n_states)
    total_supply = np.zeros(n_states)

    for tau in range(n_types):
        # Obtenemos las probabilidades para este tipo
        omega_t, p_scrap_t, p_keep_t = calc_probs(EV, p_guess, tau)
        
        # Obtenemos la distribución estacionaria q_tau = q_tau @ (omega @ Q_a)
        q_t = get_q_tau(omega_t)
        
        # --- DEMANDA AGREGADA ---------
        # Representa a dónde quieren saltar los agentes de este tipo
        # Es q_t @ omega_t: un vector de n_states con la demanda de cada bien
        prob_trade_matrix = omega_t.copy()
        np.fill_diagonal(prob_trade_matrix, prob_trade_matrix.diagonal() - p_keep_t)
        indices_usados = get_indices_usados()
        for s1 in indices_usados:
            demand_tau[s1] = np.sum(q_t * prob_trade_matrix[:, s1])
        
        # --- OFERTA AGREGADA -------------
        # Solo ofrecen coche los que NO se quedan con el suyo (1 - p_keep)
        # Y de esos, solo los que NO lo chatarrizan (1 - p_scrap)
        supply_tau = np.zeros(n_states)
        for s0 in range(n_states):
            if s0 == 0: continue
            if s0 in max_ages: continue   # no se vende
            supply_tau[s0] = q_t[s0] * (1 - p_keep_t[s0]) * (1 - p_scrap_t[s0])
        
        # Sumamos al total ponderando por la masa de este tipo de consumidor
        total_demand += type_mass[tau] * demand_tau
        total_supply += type_mass[tau] * supply_tau

    # --- EQUILIBRIO DE MERCADO ---
    
    # Calculamos el exceso de demanda bruto para todos los estados
    ED_full = total_demand - total_supply

    print(sum(total_demand))
    print(sum(total_supply))
    # FILTRO: Oferta perfectamente elástica para coches nuevos.
    # El optimizador solo debe ver el exceso de demanda de los coches usados.
    # Los coches nuevos y el estado 0 no entran en el ajuste de precios.
    
    # Supongamos que tienes una lista de índices que corresponden a coches usados
    # (aquellos con edad > 0, edad < max_age, y que no son el estado s=0)
    indices_usados = get_indices_usados() 
    
    # El vector que devolvemos tiene el mismo tamaño que p_guess
    return ED_full[indices_usados]



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
    print(f"Iteración: Error promedio (norma): {norma_error:.4f}")
    return error_vec

# -------------------------------------------------

print("Iniciando la búsqueda del equilibrio de mercado...")


# Scipy shit
eps = 1e-6
lb = np.ones_like(p_init) * eps
ub = np.ones_like(p_init) * 1e6

res = least_squares(
    fun=ED_wrapper,
    x0=p_init,
    bounds=(1e-6, np.inf),
    xtol=1e-8,
    ftol=1e-8,
    gtol=1e-8,
    max_nfev=200
)
if res.success:
    precios_equilibrio_usados = res.x
    print("least_squares convergió. norm error:", np.linalg.norm(ED(precios_equilibrio_usados)))
else:
    print("least_squares falló:", res.message)


# Paso 8: Generar datos con P ----------------------------------------------------



# Paso 3: Simulación ---------------------------------------------------------------



