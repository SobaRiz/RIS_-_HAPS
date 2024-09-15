# -*- coding: utf-8 -*-

import numpy as np
from Principal import calculate_data_rates_SNR
import json
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time




start_time = time.time()



# Chargement du fichier JSON
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

# Algorithme de Random Optimization
def random_optimization(system_model, num_iterations, K, NR, NH, P_max):
    best_solution = None
    best_data_rate = -np.inf # Utiliser une valeur très basse pour initialiser
    best_theta_values = []  # Stocke les meilleures valeurs de Theta
    iteration_indices = []  # Stocke les indices des itérations où une amélioration est trouvée

    # Liste pour stocker les résultats de chaque itération
    results = []


    for i in range(num_iterations):
        P, Theta, N_R_k, N_H_k = generate_random_solution(K, NR, NH, P_max)
        # candidate_solution = generate_random_solution(K, NR, P_max)
        # P, Theta = candidate_solution
        data_rate = evaluate_solution(system_model, P, Theta, N_R_k, N_H_k)

        # Apply constraints
        if np.sum(N_R_k) <= NR and np.sum(N_H_k) <= NH:
            # Évalue la solution en utilisant le modèle du système
            # data_rates = calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k)
            # data_rate = np.sum(data_rates)
            if data_rate > best_data_rate:
                best_solution = (P, Theta, N_R_k, N_H_k)
                best_data_rate = data_rate
                # best_theta_values.append(np.max(Theta))  # Enregistre la meilleure valeur de Theta à chaque itération
                # iteration_indices.append(i)
        # Enregistrer les résultats de cette itération
        results.append({
            'Iteration': i,  # Commencer à 0
            'Theta': Theta,
            'P': P,
            'N_R_k': N_R_k,
            'N_H_k': N_H_k,
            'DataRates': data_rate
        })




    # Préparation des données pour le DataFrame
    data = {
        'Iteration': [result['Iteration'] for result in results],
        'Theta': [result['Theta'] for result in results],
        'P': [result['P'] for result in results],
        'N_R_k': [result['N_R_k'] for result in results],
        'N_H_k': [result['N_H_k'] for result in results],
        'DataRates': [result['DataRates'] for result in results]
    }

    # Création du DataFrame pandas
    df = pd.DataFrame(data)

    # Transposer le DataFrame pour que les itérations soient les colonnes
    df_transposed = df.set_index('Iteration').T

    # # Calculer la moyenne des data rates
    # average_data_rates = np.mean(DataRates, axis=0)

    # # Ajouter la ligne pour la moyenne des data rates
    # df_transposed.loc['Average Data Rates'] = average_data_rates

    # Sauvegarde du DataFrame dans un fichier CSV
    df_transposed.to_excel('random_optimization_results.xlsx', index=True)
    print("Les résultats ont été sauvegardés dans 'random_optimization_results.xlsx'.")

    return best_solution, best_data_rate, best_theta_values, iteration_indices

# Paramètres du système
# system_model = load_parameters('json_datas.json')
K = system_model['K']  # Number of single-antenna ground users (UEs)
NR = system_model['NR']  # Number of reflective elements in RIS
NH = system_model['NH']  # Number of antennas at HAPS
P_max = system_model['P_max']  # Puissance de transmission maximale de chaque utilisateur --------> (W) entre 0.6 et 3
num_iterations = system_model['Nbr_It']  # Nombre d'itérations

# Exécute l'optimisation aléatoire
best_solution, best_data_rate, best_theta_values, iteration_indices = random_optimization(system_model, num_iterations, K, NR, NH, P_max)
P, Theta, N_R_k, N_H_k = best_solution
print("Meilleure solution:", best_solution)
print("Meilleur débit de données:", best_data_rate)

# Affichage des résultats après optimisation
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

# Plot des valeurs de P après optimisation
ax3.bar(range(1, system_model['K'] + 1), P, width=0.4, align='center', label='Meilleures valeurs de P')
ax3.set_xlabel('Utilisateur')
ax3.set_ylabel('Puissance de transmission')
ax3.set_title('Meilleures valeurs de P après optimisation')
ax3.legend()

# =============================================================================
# 2D REPRESENTATION
# Plot des valeurs de Theta après optimisation
ax4.plot(range(1, system_model['NR'] + 1), Theta, marker='o', linestyle='-', color='b', label='Meilleures valeurs de Theta')
ax4.set_xlabel('Reflective elements')
ax4.set_ylabel('Phase shift (radians)')
ax4.set_title('Meilleures valeurs de Theta après optimisation - all random')
ax4.legend()

# =============================================================================
# 3D REPRESENTATION
# =============================================================================
# # Créer des coordonnées pour chaque élément réfléchissant
# x = np.arange(1, NR + 1)
# y = np.zeros_like(x)  # Juste pour une visualisation plus simple
# z = Theta  # Valeurs de Theta en radians
# 
# # Affichage 3D
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# 
# # Scatter plot
# sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
# 
# # Ajouter une barre de couleur pour représenter les valeurs
# fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
# 
# ax.set_xlabel('Reflective elements')
# ax.set_ylabel('Dummy Axis (Y)')
# ax.set_zlabel('Phase shift (radians)')
# ax.set_title('3D Scatter Model for Theta')
# =============================================================================

plt.tight_layout()
plt.show()


# =============================================================================
# TIME
# =============================================================================
end_time = time.time()

# Create new file to put information on it
with open("Random_Opt_time.txt", "w") as file:
    file.write(f"{end_time}")
