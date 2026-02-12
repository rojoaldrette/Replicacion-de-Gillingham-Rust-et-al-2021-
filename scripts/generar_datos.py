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
        precios = np.load("input/precios_optimizados.npy")
        return precios
    except FileNotFoundError:
        print("Error: No existe el archivo. Corriendo la simulación.")
        precios_optimos = sim.ejecutar_optimizacion()
        return precios_optimos

precios = cargar_precios()


def gen_datos(precios_usados, N, T, rand_p = True):

    P = sim.get_P(precios_usados)
    q_final, q_ss, p_trade, omegas = sim.get_info(precios_usados)

    # ---- Inicializar hogares -------
    tipos = np.random.choice([0, 1], size=N, p=sim.type_mass)
    estados = np.array([np.random.choice(sim.n_states, p=q_ss[t]) for t in tipos])
    
    rows_ownership = []
    rows_transactions = []
    rows_owner_trans = []
    for t in range(T):
        for i in range(N):
            s0 = estados[i]
            tau = tipos[i]
            age_0 = sim.get_age(s0)
            brand_0 = sim.get_brand(s0)

            s1 = np.random.choice(sim.n_states, p=omegas[tau][s0])

            age_1 = sim.get_age(s1)

            if s1 != s0 and s1 != 0:
                # Precio observado = Precio equilibrio * error de calidad
                p_base = P[s1] if s1 > 0 else 0
                if rand_p:
                    p_obs = p_base * np.exp(np.random.normal(0, 0.05))
                else:
                    p_obs = p_base
                
                brand_1 = sim.get_brand(s1)
                rows_transactions.append({
                    'year': t, 'tau': tau, 'hh_id': i, 'car_id': brand_1,
                    'age': age_1, 'price': p_obs,
                })
                rows_owner_trans.append({
                    'year': t, 'tau': tau, 'hh_id': i, 'car_id_s': brand_0,
                    'age_s': age_0, 'car_id_d': brand_1,
                    'age_d': age_1, 'price': p_obs,
                })

            s2 = np.random.choice(sim.n_states, p=sim.Q_a[s1])
            estados[i] = s2
            age_2 = sim.get_age(s2)
            brand_2 = sim.get_brand(s2)
            

            rows_ownership.append({
                'year': t, 'tau': tau, 'hh_id': i, 'type': tau,
                'car_brand': brand_2, 'age': age_2
            })

    return pd.DataFrame(rows_ownership), pd.DataFrame(rows_transactions), pd.DataFrame(rows_owner_trans)


if __name__ == "__main__":
    df_owner, df_trans, df_owner_trans = gen_datos(precios, 10000, 7)
    df_owner.to_csv('output/df_owner.csv')
    df_trans.to_csv('output/df_trans.csv')
    df_owner_trans.to_csv('output/df_owner_trans.csv')


    df_owner[(df_owner["hh_id"] == 5)]
