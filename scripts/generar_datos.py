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


def gen_datos(precios_usados, N, T, rand_p = True):

    P = sim.get_P(precios_usados)
    q_final, q_ss, p_trade, omegas = sim.get_info(precios_usados)

    # ---- Inicializar hogares -------
    tipos = np.random.choice([0, 1], size=N, p=sim.type_mass)
    estados = np.array([np.random.choice(sim.n_states, p=q_ss[t]) for t in tipos])

    for t in range(T):
        for i in range(N):
            s0 = estados[i]
            tau = tipos[i]

            rows_ownership = []
            rows_transactions = []

            s1 = np.random.choice(sim.n_states, p=omegas[tau][s0])

            age = sim.get_age(s1)
            if s1 != s0 and s1 != 0:
                # Precio observado = Precio equilibrio * error de calidad
                p_base = P[s1] if s1 > 0 else 0
                if rand_p:
                    p_obs = p_base * np.exp(np.random.normal(0, 0.05))
                else:
                    p_obs = p_base
                
                rows_transactions.append({
                    'year': t, 'hh_id': i, 'car_id': s1,
                    'price': p_obs, 'brand': sim.get_brand(s1),
                    'age': age # En el momento de compra siempre es el 'nuevo' del dueño
                })

            s2 = np.random.choice(sim.n_states, p=sim.Q_a[s1])
            estados[i] = s2
            age = sim.get_age(s2)
            brand = sim.get_brand(s2)
            
            # 4. Registro de tenencia (Ownership)
            rows_ownership.append({
                'year': t, 'hh_id': i, 'type': tau, 'car_brand': brand, 'age': age
            })

    return pd.DataFrame(rows_ownership), pd.DataFrame(rows_transactions)


if __name__ == "__main__":
    #df_owner, df_trans = gen_datos(precios, 1000, 7)
    q_final, q_ss, p_trade, omegas = sim.get_info(precios)

q_test = sim.get_q_tau(omegas[0])

if not np.allclose(omegas[0].sum(axis=1), 1):
    print("nel we")
