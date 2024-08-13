# -*- coding: utf-8 -*-

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from Principal import calculate_data_rates_SNR, negative_sum_rate_theta_fixed_SNR

# Charger le modèle de système à partir d'un fichier JSON
with open('json_datas.json', 'r') as f:
    system_model = json.load(f)

# Génère une solution aléatoire pour le modèle du système
def generate_random_solution(K, NR, NH, P_max):
    P = np.random.uniform(0.1, P_max, K)
    Theta = np.random.uniform(0, 2 * np.pi, NR)
    N_R_k = np.random.randint(1, NR + 1, K)
    N_H_k = np.random.randint(1, NH + 1, K)
    return P, Theta, N_R_k, N_H_k

# Évalue la solution en utilisant le modèle du système
def evaluate_solution(system_model, P, Theta, N_R_k, N_H_k):
    return calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k)

# Algorithme de l'optimisation par les baleines
def whale_optimization(system_model, num_iterations, num_whales, K, NR, NH, P_max):
    # Initialisation de la population de baleines
    whales = [generate_random_solution(K, NR, NH, P_max) for _ in range(num_whales)]
    best_solution = None
    best_data_rate = -np.inf

    # Initialisation d'une liste pour stocker les résultats de chaque itération
    results_list = []


    for i in range(num_iterations):
        for whale in whales:
            P, Theta, N_R_k, N_H_k = whale
            data_rate = evaluate_solution(system_model, P, Theta, N_R_k, N_H_k)
            
            if np.sum(N_R_k) <= NR and np.sum(N_H_k) <= NH and np.sum(data_rate) > best_data_rate:
                best_solution = whale
                best_data_rate = np.sum(data_rate)

        # Mettre à jour les baleines en fonction de la meilleure baleine
        for j in range(num_whales):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * r1 - 1
            C = 2 * r2
            best_P, best_Theta, best_N_R_k, best_N_H_k = best_solution
            current_P, current_Theta, current_N_R_k, current_N_H_k = whales[j]

            D_P = np.abs(C * best_P - current_P)
            D_Theta = np.abs(C * best_Theta - current_Theta)
            D_N_R_k = np.abs(C * best_N_R_k - current_N_R_k)
            D_N_H_k = np.abs(C * best_N_H_k - current_N_H_k)

            whales[j] = (
                best_P - A * D_P,
                best_Theta - A * D_Theta,
                best_N_R_k - A * D_N_R_k,
                best_N_H_k - A * D_N_H_k
            )

        # Créer un dictionnaire pour stocker les résultats
        results = {
            'Iteration': i + 1,
            'P': best_solution[0],
            'DataRates': best_data_rate
        }
        
        # Ajouter les informations de Theta
        for idx, theta_val in enumerate(best_solution[1]):
            results[f'Theta_{idx}'] = theta_val

        # Ajouter les informations de N_R_k
        for idx, n_r_k_val in enumerate(best_solution[2]):
            results[f'N_R_k_{idx}'] = n_r_k_val

        # Ajouter les informations de N_H_k
        for idx, n_h_k_val in enumerate(best_solution[3]):
            results[f'N_H_k_{idx}'] = n_h_k_val

        # Enregistrer les résultats
        results_list.append(results)

        # # Enregistrer les résultats
        # results.append({
        #     'Iteration': i + 1,
        #     'Theta': best_solution[1],
        #     'P': best_solution[0],
        #     'N_R_k': best_solution[2],
        #     'N_H_k': best_solution[3],
        #     'DataRates': best_data_rate
        # })

    return best_solution, best_data_rate, results_list


# Exécution de l'optimisation par les baleines
num_whales = 30
num_iterations = system_model['Nbr_It']
K = system_model['K']
B = system_model['B']
N0 = system_model['N0']
NR = system_model['NR']
NH = system_model['NH']
P_max = system_model['P_max']

best_solution, best_data_rate, results = whale_optimization(system_model, num_iterations, num_whales, K, NR, NH, P_max)
# best_solution, best_data_rate = whale_optimization(system_model, num_iterations, num_whales, K, NR, NH, P_max)
P, Theta, N_R_k, N_H_k = best_solution
# print(type(results[Theta]), type(results[P]), type(results[N_R_k]), type(results[N_H_k]), type(results[best_data_rate]))


# Préparation des données pour le DataFrame
# data = {
#     'Iteration': [result['Iteration'] for result in results],
#     'Theta': [result['Theta'] for result in results],
#     'P': [result['P'] for result in results],
#     'N_R_k': [result['N_R_k'] for result in results],
#     'N_H_k': [result['N_H_k'] for result in results],
#     'DataRates': [result['DataRates'] for result in results]
# }


# Création du DataFrame
df = pd.DataFrame(results)
print(results)

# Sauvegarde dans un fichier Excel
df.to_excel('whales_optimization_results.xlsx', index=False)

# Ajouter une ligne pour la moyenne des DataRates
average_data_rates = df['DataRates'].mean()
df.loc[len(df.index)] = ['Average'] + [None] * (df.shape[1] - 2) + [average_data_rates]

print("Les résultats ont été sauvegardés dans 'whales_optimization_results.xlsx'.")



# # Représentation graphique de Theta au cours des itérations
# plt.figure(figsize=(10, 6))
# for i, result in enumerate(results):
#     plt.plot(result['Theta'], label=f'Iteration {result["Iteration"]}')

# plt.xlabel('Element Index')
# plt.ylabel('Theta (radians)')
# plt.title('Evolution of Theta Over Iterations')
# plt.legend()

# Représentation graphique de Theta après optimisation
# best_theta = results[-1]['Theta']
fig, ax4 = plt.subplots(1, 1, figsize=(12, 6))

ax4.plot(range(1, system_model['NR'] + 1), Theta, marker='o', linestyle='-', color='b', label='Meilleures valeurs de Theta')
ax4.set_xlabel('Reflective elements')
ax4.set_ylabel('Phase shift (radians)')
ax4.set_title('Meilleures valeurs de Theta après optimisation')
ax4.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# TEST - Distribution des décalages de phase par utilisateur
cumulative_index = 0
for k, n_r_k in enumerate(best_solution[2]):  # best_solution[2] corresponds to N_R_k
    n_r_k = int(n_r_k)  # Conversion en entier
    end_index = cumulative_index + n_r_k
    plt.plot(range(cumulative_index + 1, end_index + 1), best_solution[1][cumulative_index:end_index], 
             marker='o', linestyle='-', label=f'Utilisateur {k + 1}')
    cumulative_index = end_index

plt.xlabel('Éléments réflecteurs')
plt.ylabel('Phase shift (Theta) en radians')
plt.title('Distribution des décalages de phase par utilisateur')
plt.legend()
plt.show()
# =============================================================================

# =============================================================================
# TEST - Évolution des débits de données au cours des itérations
iterations = [result['Iteration'] for result in results]
data_rates = [result['DataRates'] for result in results]

plt.plot(iterations, data_rates, marker='o', linestyle='-', color='g', label='Débits de données')
plt.xlabel('Itérations')
plt.ylabel('Débits de données (Mbps)')
plt.title('Évolution des débits de données au cours des itérations')
plt.legend()
plt.show()
# =============================================================================
