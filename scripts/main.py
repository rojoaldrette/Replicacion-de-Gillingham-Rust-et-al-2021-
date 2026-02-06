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
# actualización:  29/01/2026
#
# _____________________________________________________________________________

# PREAMBULO ___________________________________________________________________
import pandas as pd
import numpy as np

# CODIGO ______________________________________________________________________

# FUNCIONES -------------------------------------------------------------------

# DATA ------------------------------------------------------------------------

# SCRIPT ----------------------------------------------------------------------


def diagnostico_oferta(p_guess):
    EV = sol_bellman(p_guess)
    for tau in range(n_types):
        omega_t, p_scrap_t, p_keep_t = calc_probs(EV, p_guess, tau)
        q_t = get_q_tau(omega_t)

        # Agregados simples
        total_keep_prob = np.sum(q_t * p_keep_t)                 # masa que se queda
        total_trade_prob = np.sum(q_t * (1 - p_keep_t))          # masa que no se queda (trade o scrap)
        total_scrap_prob = np.sum(q_t * (1 - p_keep_t) * p_scrap_t)  # masa que efectivamente chatarriza
        total_offer_prob = np.sum(q_t * (1 - p_keep_t) * (1 - p_scrap_t))  # oferta por estado

        print(f"--- Tipo {tau} ---")
        print("Masa total (q_t sum):", np.sum(q_t))
        print("Quedan (sum q*p_keep):", total_keep_prob)
        print("Deciden no quedarse (sum q*(1-p_keep)):", total_trade_prob)
        print("De esos, chatarrizan (sum q*(1-p_keep)*p_scrap):", total_scrap_prob)
        print("De esos, ofrecen (sum q*(1-p_keep)*(1-p_scrap)):", total_offer_prob)

        # Distribución por edad (promedio ponderado de prob de ofertar por edad)
        ages = np.array([get_age(s) for s in range(n_states)])
        brands = np.array([get_brand(s) for s in range(n_states)])
        used_indices = get_indices_usados()

        # Por edad promedio oferta
        print("Oferta por edad (edad, oferta_mass):")
        for a in range(1, A_max+1):
            idxs = [s for s in used_indices if get_age(s) == a]
            if len(idxs) == 0: 
                continue
            offer_mass = np.sum([q_t[s] * (1 - p_keep_t[s]) * (1 - p_scrap_t[s]) for s in idxs])
            print(a, offer_mass)
        print("-----\n")

diagnostico_oferta(p_init)

EV_test = sol_bellman(p_init)
total_demand = np.zeros(n_states)
total_supply = np.zeros(n_states)
for tau in range(n_types):
    omega_t, p_scrap_t, p_keep_t = calc_probs(EV_test, p_init, tau)
    q_t = get_q_tau(omega_t)
    demand_tau = q_t @ omega_t
    supply_tau = np.zeros(n_states)
    for s0 in range(n_states):
        if s0 == 0: continue
        if s0 in max_ages: continue
        supply_tau[s0] = q_t[s0] * (1 - p_keep_t[s0]) * (1 - p_scrap_t[s0])
    total_demand += type_mass[tau] * demand_tau
    total_supply += type_mass[tau] * supply_tau

print("SUM DEMAND:", np.sum(total_demand))
print("SUM SUPPLY:", np.sum(total_supply))


