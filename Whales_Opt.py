# -*- coding: utf-8 -*-

# Code permettant de mettre en place la méthode Whales

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from Principal import calculate_data_rates_SNR, negative_sum_rate_theta_fixed_SNR

# =============================================================================
# Ce code sert à mettre en place la méthode d'optimisation des baleines avec les composants SNR présent dans le fichier "Principal.py"

# MOTS CLES (définition à retrouver dans le READ.ME):
# ----------------------------------------------------
    # Whales optimisation
    # SNR (Signal-to-Noise Ratio)
    # Theta (phases ajustables)
    # RIS (Reconfigurable Intelligent Surfaces)
    # éléments RIS (éléments dans une surface intelligente reconfigurable)
    # HAPS (High Altitude Platform Station)
    # débits de données
    # Phase shift (décalage de phase en radians)
    # Population de baleines
    # NR (nombre d'éléments réfléchissants dans le RIS)
    # NH (nombre d'antennes à HAPS)
    # P_max (puissance maximale de transmission pour chaque utilisateur)
    # Meilleures valeurs de P (puissances optimisées)
# =============================================================================

# Charger le modèle de système à partir d'un fichier JSON
with open('json_datas.json', 'r') as f:
    system_model = json.load(f)

# Évalue la solution en utilisant le modèle du système
def evaluate_solution(system_model, P, Theta, N_R_k, N_H_k):
    return calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k)

# Génère une solution aléatoire pour le modèle du système
def generate_random_solution(K, NR, NH, P_max):
    P = np.random.uniform(0.1, P_max, K)  # Génération de puissances aléatoires entre 0.1 et P_max pour chaque utilisateur
    Theta = np.random.uniform(0, 2 * np.pi, NR)  # Génération de décalages de phase aléatoires pour chaque élément réflecteur
    N_R_k = np.random.randint(1, NR + 1, K)  # Attribution aléatoire du nombre d'éléments réflecteurs à chaque utilisateur
    N_H_k = np.random.randint(1, NH + 1, K)  # Attribution aléatoire du nombre d'éléments d'antenne à chaque utilisateur
    return P, Theta, N_R_k, N_H_k

# =============================================================================
# Algorithme de l'optimisation par les baleines
# =============================================================================
def whale_optimization(system_model, num_iterations, num_whales, K, NR, NH, P_max):
    # Initialisation de la population de baleines avec des solutions aléatoires
    whales = [generate_random_solution(K, NR, NH, P_max) for _ in range(num_whales)]
    best_solution = None
    best_data_rate = -np.inf  # Débit de données initial le plus bas

    # Initialisation d'une liste pour stocker les résultats de chaque itération
    results_list = []

    a = 0

    for i in range(num_iterations):
        for whale in whales:
            P, Theta, N_R_k, N_H_k = whale
            data_rate = evaluate_solution(system_model, P, Theta, N_R_k, N_H_k)
            
            # Vérifie si la solution est meilleure que la meilleure solution actuelle
            if np.sum(N_R_k) <= NR and np.sum(N_H_k) <= NH and np.sum(data_rate) > best_data_rate:
                best_solution = whale
                best_data_rate = np.sum(data_rate)

        # Mise à jour des solutions des baleines en fonction de la meilleure solution actuelle
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
        a =a+1
        # os.system('cls')
        print("Itération",a)

        # Stocke les résultats de l'itération actuelle
        results = {
            'Iteration': i + 1,
            'P': best_solution[0],
            'DataRates': best_data_rate
        }
        
        # Ajoute les valeurs de Theta pour chaque élément réflecteur
        for idx, theta_val in enumerate(best_solution[1]):
            results[f'Theta_{idx}'] = theta_val

        # Ajoute les valeurs de N_R_k et N_H_k pour chaque utilisateur
        for idx, n_r_k_val in enumerate(best_solution[2]):
            results[f'N_R_k_{idx}'] = n_r_k_val

        for idx, n_h_k_val in enumerate(best_solution[3]):
            results[f'N_H_k_{idx}'] = n_h_k_val

        # Ajoute les résultats à la liste
        results_list.append(results)

    return best_solution, best_data_rate, results_list

# =============================================================================
# Exécution de l'optimisation
# =============================================================================

num_whales = 30
num_iterations = system_model['Nbr_It']
K = system_model['K']
B = system_model['B']
N0 = system_model['N0']
NR = system_model['NR']
NH = system_model['NH']
P_max = system_model['P_max']

best_solution, best_data_rate, results = whale_optimization(system_model, num_iterations, num_whales, K, NR, NH, P_max)
P, Theta, N_R_k, N_H_k = best_solution

# Création du DataFrame à partir des résultats
df = pd.DataFrame(results)
print(results)

# Sauvegarde des résultats dans un fichier Excel
df.to_excel('whales_optimization_results.xlsx', index=False)

# Ajout d'une ligne pour la moyenne des DataRates
average_data_rates = df['DataRates'].mean()
df.loc[len(df.index)] = ['Average'] + [None] * (df.shape[1] - 2) + [average_data_rates]

print("Les résultats ont été sauvegardés dans 'whales_optimization_results.xlsx'.")

# =============================================================================
# REPRESENTATIONS
# =============================================================================

# # Plot des valeurs de P après optimisation
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, K + 1), P, marker='o', linestyle='-', color='b', label='Valeurs initiales de P')
# plt.xlabel('Utilisateur')
# plt.ylabel('Puissance de transmission')
# plt.title('Meilleures valeurs de P après optimisation')
# plt.legend()


# Représentation graphique des meilleures valeurs de Theta après optimisation
fig, ax4 = plt.subplots(1, 1, figsize=(12, 6))

ax4.plot(range(1, system_model['NR'] + 1), Theta, marker='o', linestyle='-', color='b', label='Meilleures valeurs de Theta')
ax4.set_xlabel('Éléments réflecteurs')
ax4.set_ylabel('Phase shift (radians)')
ax4.set_title('Meilleures valeurs de Theta après optimisation')
ax4.legend()

plt.tight_layout()
plt.show()

# Distribution des décalages de phase par utilisateur
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

# Évolution des débits de données au cours des itérations
iterations = [result['Iteration'] for result in results]
data_rates = [result['DataRates'] for result in results]

plt.plot(iterations, data_rates, marker='o', linestyle='-', color='g', label='Débits de données')
plt.xlabel('Débits de données (Mbps)')
plt.ylabel('Itérations')
plt.title('Évolution des débits de données au cours des itérations')
plt.legend()
plt.show()

# Représentation graphique des sum data rates au fil des itérations
plt.figure(figsize=(10, 6))
plt.plot(iterations, data_rates, marker='o', linestyle='-', color='purple', label='Sum Data Rates')
plt.xlabel('Sum Data Rates (Mbps)')
plt.ylabel('Iterations')
plt.title('Sum Data Rates au fil des itérations')
# plt.gca().invert_yaxis()  # Inverser l'axe des y pour que les itérations augmentent vers le haut
plt.legend()
plt.show()
# =============================================================================
