import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
import random
import sympy as sp
import matplotlib.pyplot as plt
import json

# Chargement du fichier JSON
with open('json_datas.json', 'r') as f:
    parameters = json.load(f)
    print(parameters)



# =============================================================================
# CONSTANTES ET PARAMETRES
# =============================================================================
# Define system parameters and constants
K = parameters['K']  # Number of single-antenna ground users (UEs)
NR = parameters['NR']  # Number of reflective elements in RIS
NH = parameters['NH']  # Number of antennas at HAPS
B = parameters['B']  # Total available bandwidth (en Mhz)  -----> (devient 20e6 Hz)
N0 = parameters['N0']  # Puissance du bruit blanc additif (AWGN) ------> (en dB)
R_th = parameters['R_th']  # Débit de données minimal pour chaque utilisateur --------> (bit/s)
P_max = parameters['P_max']  # Puissance de transmission maximale de chaque utilisateur --------> (W) entre 0.6 et 3


# Define channel parameters     ------>entiere
m = parameters['m']  # Nakagami-m fading parameter
GH = parameters['GH']  # Antenna gain of HAPS -------->  (db)
c = parameters['c']  # Speed of light ------> (m/s)
dRH = parameters['dRH']  # Distance between RIS and HAPS ------> (m)
fc = parameters['fc']  # Carrier frequency -----> (Hz)  (2(better for the coverage) or 6 Ghz)

# =============================================================================
# INITIALISATION DES VARIABLES
# =============================================================================

# Initialisation de la configuration de phase-shift
def initialisation_variables(np, random):
    Theta = np.random.uniform(0, 2 * np.pi, NR)
    return Theta
Theta = initialisation_variables(np, random)


def set_haps_coordinates():
    print("Enter coordinates for HAPS:")
    # haps_x = float(input("x-coordinate: "))
    # haps_y = float(input("y-coordinate: "))
    # haps_z = float(input("z-coordinate: "))

    haps_x = 0
    haps_y = 0
    haps_z = 20000

    return haps_x, haps_y, haps_z
# Utilisez cette fonction pour obtenir les coordonnées de HAPS
haps_coordinates = set_haps_coordinates()


print("HAPS coordinates:", haps_coordinates)


def get_ris_coordinates(max_coordinate):
    print("Enter coordinates for RIS:")
    while True:
        # ris_x = float(input("x-coordinate: "))
        # ris_y = float(input("y-coordinate: "))
        # ris_z = float(input("z-coordinate: "))

        ris_x = 30
        ris_y = 300
        ris_z = 1000

        if 0 <= ris_x <= max_coordinate and 0 <= ris_y <= max_coordinate and 0 <= ris_z <= 20000:
            return ris_x, ris_y, ris_z
        else:
            print(f"Coordinates out of bounds. Please make sure x and y are within [0, {max_coordinate}].")
max_coordinate = 500000
ris_coordinates = get_ris_coordinates(max_coordinate)
print("RIS coordinates:", ris_coordinates)


def get_user_coordinates(K, max_coordinate):
    ris_x, ris_y, ris_z = get_ris_coordinates(max_coordinate)
    coordinates = [(ris_x, ris_y, ris_z)]  # Ajoutez les coordonnées du RIS à la liste
    for i in range(K):
        print(f"Enter coordinates for user {i + 1}:")
        while True:
            # x = float(input("x-coordinate: "))
            # y = float(input("y-coordinate: "))
            # z = float(input("z-coordinate: "))

            x = 1
            y = 1
            z = 1
            if 0 <= x <= max_coordinate and 0 <= y <= max_coordinate and 0 <= z <= ris_z:
                coordinates.append((x, y, z))
                break
            else:
                print(
                    f"Coordinates out of bounds. Please make sure x, y, and z are within [0, {max_coordinate}] for x and y, and within [0, {ris_z}] for z.")
    return coordinates
# Utilisez cette fonction pour obtenir les coordonnées des utilisateurs
max_coordinate = 500000
user_coordinates = get_user_coordinates(K, max_coordinate)
print("User coordinates:", user_coordinates)


# =============================================================================
# CALCUL DES DISTANCES ET DES ANGLES
# =============================================================================

# Utilisation de Thalès et Pythagore pour pouvoir connaitre l'hypothénuse et angles correspondants à chaque triangle rencontré
#        +++
#        +++       Avec angles : %
#       /%|        (a) Distance RIS : |
#      /  |        (b) Distance User-Sol : [0,0]
#     /   |        Distance UR : ||a-b||
#    /    |
#   /     |
#  /      |
# o%______|
# |
# ^
def calculate_distances_and_angles(user_coordinates, ris_coordinates, haps_coordinates):
    print("thabet fil user coord", user_coordinates)
    print("thabet fil ris coord", ris_coordinates)
    print("thabet fil haps coord", haps_coordinates)
    distances = []
    angle_phi_DU = []
    angle_phi_AR = []
    angle_phi_DR = []
    angle_phi_AH = []
    d_UR_list = []
    d_RH_list = []
    for i in range(1, K + 1):
        # Distances and angles with respect to RIS
        dis_RIS_U = abs(ris_coordinates[0] - user_coordinates[i][0])
        h_RIS_U = ris_coordinates[2] - user_coordinates[i][2]
        d_UR = np.sqrt(dis_RIS_U ** 2 + h_RIS_U ** 2)
        if d_UR == 0:
            phi_DU = 0
            phi_AR = 0
        else:
            phi_DU = np.arccos(dis_RIS_U / d_UR)
            phi_AR = np.arccos(h_RIS_U / d_UR)

        # Distances and angles with respect to HAPS
        dis_HAPS_RIS = abs(ris_coordinates[0] - haps_coordinates[0])
        h_HAPS_RIS = haps_coordinates[2] - ris_coordinates[2]
        d_RH = np.sqrt(h_HAPS_RIS ** 2 + dis_HAPS_RIS ** 2)
        if d_RH == 0:
            phi_DR = 0
            phi_AH = 0
        else:
            phi_DR = np.arccos(dis_HAPS_RIS / d_RH)
            phi_AH = np.arccos(h_HAPS_RIS / d_RH)
        d_UR_list.append(d_UR)
        d_RH_list.append(d_RH)
        distances.append((dis_RIS_U, h_RIS_U, d_UR, dis_HAPS_RIS, h_HAPS_RIS, d_RH))
        angle_phi_DU.append(phi_DU)
        angle_phi_AR.append(phi_AR)
        angle_phi_DR.append(phi_DR)
        angle_phi_AH.append(phi_AH)
    return distances, angle_phi_DU, angle_phi_AR, angle_phi_DR, angle_phi_AH, d_UR_list, d_RH_list
distances, angle_phi_DU, angle_phi_AR, angle_phi_DR, angle_phi_AH, d_UR_list, d_RH_list = calculate_distances_and_angles(
    user_coordinates, ris_coordinates, haps_coordinates)
for i in range(K):
    print(f"Distances and angles for user {i + 1}:")
    print("Distances between user and RIS:", d_UR_list[i])
    print("Distances between RIS and HAPS:", d_RH_list[i])
    print("Angle phi_DU", angle_phi_DU[i])
    print("Angle phi_AR", angle_phi_AR[i])
    print("Angle phi_DR", angle_phi_DR[i])
    print("Angle phi_AH", angle_phi_AH[i])

print("+++++++++")
print("+++++++++")
print("+++++++++")
print("+++++++++")
print("+++++++++")
print("+++++++++")

# =============================================================================
# GENERATION VALEUR ALPHA SELON DISTRIBUTION NAKAGAMI-m
# =============================================================================

def nakagami_m(m, NH, NR):
    n = np.zeros((NH, NR))
    for i in range(int(2 * m)):
        n += np.random.randn(NH, NR) ** 2
    n /= (2 * m)
    phi = 2 * np.pi * np.random.rand(NH, NR)
    alpha = np.sqrt(n) * np.cos(phi) + 1j * np.sqrt(n) * np.sin(phi)
    return alpha


alpha = nakagami_m(m, NH, NR)
L = alpha.size  # Update L to match the size of alpha


# =============================================================================
# CALCUL DES MATRICES DE CANAUX (H_UR & H_RH)
# =============================================================================

# Formule 2
# Définition de la fonction pour calculer la matrice g(x, φ)
def calculate_g(x, phi):
    return ((1 / np.sqrt(x)) * np.array([np.exp(1j * l * np.pi * np.cos(phi)) for l in range(x)]))


# Formule 1
# HU R =√ NRKL (L∑l)=1αldU R,l(NR, φAR,l) [g(K, φDU ,l)]T
def calculate_hur(alpha, angle_phi_DU, angle_phi_AR, d_UR_list):
    HUR = np.zeros((NR, K), dtype=complex)
    for k in range(K):
        HUR_k = np.zeros((NR, 1), dtype=complex)
        for l in range(min(alpha.shape[0], alpha.shape[1])):  # Adjust the loop to iterate over the correct dimensions
            g_NRAR = calculate_g(NR, angle_phi_AR[k]).reshape(NR, 1)
            g_1DU_T = calculate_g(1, angle_phi_DU[k]).conj().T

            alpha_l = alpha[l, :].reshape(NR, 1) / d_UR_list[k]
            HUR_k += alpha_l * np.outer(g_NRAR, g_1DU_T)
        HUR_k *= np.sqrt(NR / L)
        HUR[:, k] = HUR_k.conj().T
    return HUR

# Formule 3
# HRH = √NRNH PL g(NH , φAH ) [g(NR, φDR)]T
# Avec PL = GH * (c / (4 * np.pi * dRH * fc)) ** 2)
# Function to calculate RIS-HAPS uplink channel matrix
def calculate_hrh(angle_phi_AR, angle_phi_DU):
    # Suppose Theta is an array of phase shifts
    HRH = np.zeros((NH, NR), dtype=complex)

    g_NHAH = calculate_g(NH, angle_phi_AR[0])
    g_NRDU_T = calculate_g(NR, angle_phi_DU[0]).conj().T  # No need for transpose and conjugate

    HRH = np.sqrt(NR * NH * GH * (c / (4 * np.pi * dRH * fc)) ** 2) * np.outer(g_NHAH, g_NRDU_T)
    return HRH

H_RH = calculate_hrh(angle_phi_AR, angle_phi_DU)
H_UR = calculate_hur(alpha, angle_phi_AR, angle_phi_DU, d_UR_list)


# =============================================================================
# CALCUL DES DEBITS DE DONNEES
# =============================================================================

# γk = Pk||HkRH ΘkhkU R|| / BkN0, ∀k = 1, . . . , K
def calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k):
    # Assuming H_RH and H_UR are part of the system_model
    # H_RH = np.random.randn(NR, NH)  # Placeholder for actual H_RH from the system model
    # H_UR = np.random.randn(NH, K)   # Placeholder for actual H_UR from the system model

    # # On s'assure que les dimensions correspondent
    # assert H_RH.shape == (NR, NH), f"H_RH dimensions should be ({NR}, {NH}), but are {H_RH.shape}"
    # assert H_UR.shape == (NH, K), f"H_UR dimensions should be ({NH}, {K}), but are {H_UR.shape}"

    # SNR=Psignal/Pnoise
    data_rates_SNR = np.zeros(K)
    H_combined = np.dot(np.dot(H_RH, np.exp(1j * np.diag(Theta))), H_UR)

    for k in range(K):

        # # Calculer la combinaison des matrices canaux avec Theta
        # H_combined_k = np.dot(np.dot(H_RH_k, np.exp(1j * np.diag(Theta[:N_R_k[k]]))), H_UR_k)

        # Calculer le signal pour l'utilisateur k
        signal = P[k] * np.linalg.norm(H_combined[:, k])**2

        # Calculer le bruit
        noise = N0 * B
        
        # Calculer le SNR
        SNR = signal / noise
        
        # Calculer le débit de données pour l'utilisateur k
        data_rates_SNR[k] = B * np.log2(1 + SNR)





        # signal = P[k] * np.linalg.norm(H_combined[:, k])**2

        # # Add white noise who cannot be erase
        # noise = N0 * B
        # SNR = signal / noise

        # # Debugging statement to check the SNR value
        # # print(f"SNR for user {k}: {SNR} (type: {type(SNR)}, shape: {np.shape(SNR)})")

        # # # Ensure SNR is a scalar
        # # if np.isscalar(SNR):
        # data_rates_SNR[k] = B * np.log2(1 + SNR)
        # # else:
        # #     raise ValueError(f"SNR for user {k} is not a scalar. Value: {SNR}")
    return np.sum(data_rates_SNR)

# γ′k = Pk|| (HRH ΘHU R)k ||²/∑Kj=k+1 Pj || (HRH ΘHU R)j ||² + BN0, ∀k = 1, . . . , K − 1,
# γ′K = PK || (HRH ΘHU R)K ||²/BN0
def calculate_data_rates_SINR(P, Theta):
    # SINR=​Psignal​/Pinterference​+Pnoise
    data_rates_SINR = np.zeros(K)
    H_combined = np.dot(np.dot(H_RH, np.exp(1j * np.diag(Theta))), H_UR)

    for k in range(K):

        signal = P[k] * np.linalg.norm(H_combined[:, k])

        # Calculate the interference term
        interference = 0
        for j in range(K):
            if j > k:
                interference += P[j] * np.linalg.norm(H_combined[:, j])

        # Add noise to interference
        noise = N0 * B
        SINR = signal / (interference + noise)

        # Debugging statement to check the SINR value
        # print(f"SINR for user {k}: {SINR} (type: {type(SINR)}, shape: {np.shape(SINR)})")

        # Ensure SINR is a scalar
        # if np.isscalar(SINR):
        data_rates_SINR[k] = B * np.log2(1 + SINR)
        # else:
        #    raise ValueError(f"SINR for user {k} is not a scalar. Value: {SINR}")
    return data_rates_SINR


# Initialisation de la puissance de transmission
P = np.ones(K) * 75
print("+++++++++")
print("+++++++++")
print("+++++++++")
print("+++++++++")
# print("Débits de données initiaux SINR :", calculate_data_rates_SINR(P, Theta))
# print("Débits de données initiaux SNR :", calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k))
print("+++++++++")
print("+++++++++")
print("+++++++++")
print("+++++++++")

# =============================================================================
# OPTIMISATION DES PUISSANCES DE TRANSMISSION (P) ET DES PHASES (THETA)
# =============================================================================

# # Fonction objectif pour P SINR
# def negative_sum_rate_p_fixed_SINR(Theta, P):
#     data_rates_SINR = calculate_data_rates_SINR(P, Theta)
#     return - np.sum(data_rates_SINR)

# Fonction objectif pour P SNR
def negative_sum_rate_p_fixed_SNR(system_model, P, Theta, N_R_k, N_H_k):
    data_rates_SNR = calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k)
    return - np.sum(data_rates_SNR)

# # Fonction objectif pour Theta SINR
# def negative_sum_rate_theta_fixed_SINR(P, Theta):
#     data_rates_SINR = calculate_data_rates_SINR(P, Theta)
#     return - np.sum(data_rates_SINR)

# Fonction de coût pour l'optimisation
def negative_sum_rate_theta_fixed_SNR(Theta, P, system_model, N_R_k, N_H_k):
    data_rates = calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k)
    return -np.sum(data_rates)


# =============================================================================
# SINR OPTIMIZATION
# =============================================================================

# print("++++++++++++++++++++++++++++++++++++++")
# print("before optimization")
# print("P values before optimization ", P)
# print("Theta values before optimization ", Theta)
# print("rates SINR:", calculate_data_rates_SINR(P, Theta))
# print("+++++++++++++++++++++++++++++++++++++++")*

# =============================================================================
# GRAPHIQUES
# =============================================================================

# # Affichage des résultats avant optimisation
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# # Plot des valeurs de P avant optimisation
# ax1.plot(range(1, K + 1), P, marker='o', linestyle='-', color='b', label='Valeurs initiales de P')
# ax1.set_xlabel('Utilisateur')
# ax1.set_ylabel('Puissance de transmission')
# ax1.set_title('Valeurs de P avant optimisation')
# ax1.legend()

# # Plot des valeurs de Theta avant optimisation
# ax2.plot(range(1, NR + 1), Theta, marker='o', linestyle='-', color='b', label='Valeurs initiales de Theta')
# ax2.set_xlabel('Reflective elements')
# ax2.set_ylabel('Phase shift (radians)')
# ax2.set_title('Valeurs de Theta avant optimisation')
# ax2.legend()

# plt.tight_layout()
# plt.show()
# =============================================================================
# =============================================================================


# # Définir les contraintes de débit de données
# r_th = 100  # rate threshold
# min_p_max = 1e-4  # Minimum value for p_max
# max_p_max = 1e-2  # Maximum value for p_max

# # Generate random p_max values for each user
# p_max_values = [random.uniform(min_p_max, max_p_max) for _ in range(K)]

# # Initialize the bounds with random upper bounds
# bounds_p = Bounds([0 for _ in range(K)], p_max_values)
# # Define bounds for Theta
# bounds_theta = Bounds([0] * NR, [2 * np.pi] * NR)

# # premiere contrainte  Rth =< R_k

# # Définir les contraintes de débit de données
# const_theta = [{'type': 'ineq', 'fun': lambda Theta: calculate_data_rates_SINR(P, Theta)[k] - r_th} for k in range(K)]
# const_p = [{'type': 'ineq', 'fun': lambda P: calculate_data_rates_SINR(P, Theta)[k] - r_th} for k in range(K)]

# const_theta = []
# for k in range(K):
#     const_theta.append(
#         {'type': 'ineq',
#                   'fun': lambda Theta:  r_th - calculate_data_rates_SINR(P, Theta)[k],  # Ensure each data rate >= r_th
#                   "keep_feasible": True
#                   }
#     )

# const_p = []
# for k in range(K):
#     const_p.append(
#         {'type': 'ineq',
#                   'fun': lambda P:  r_th - calculate_data_rates_SINR(P, Theta)[k],  # Ensure each data rate >= r_th
#                   "keep_feasible": True
#                   }
#     )



# """
# # deuxieme contrainte  0 =< P_k =< P_k_max
# ineq_cons_p = {'type': 'ineq',
#                 'fun': lambda P, P_max: P_max - P,  # Ensure each P_k <= P_max[k]
#                 'args': (p_max_values,),
#                 "keep_feasible": True
#                 }

# # troixieme contrainte  0 =< Theta_k =< 2*pi
# ineq_cons_theta = {'type': 'ineq',
#                     'fun': lambda Theta_flat: np.concatenate((Theta_flat.reshape(NR, NR).flatten(),
#                                                               np.pi - Theta_flat.reshape(NR, NR).flatten(),
#                                                               -Theta_flat.reshape(NR, NR).flatten())),
#                     "keep_feasible": True
#                     }
# """

# # Optimize p given fixed theta for SINR
# res_p = minimize(negative_sum_rate_theta_fixed_SNR,
#                   P,
#                   method='SLSQP',
#                   args=(Theta,),
#                   constraints=const_p,
#                   bounds=bounds_p,
#                   options={'disp': True, "maxiter": 100000})

# P = res_p.x

# res_Theta = minimize(negative_sum_rate_p_fixed_SNR,
#                       Theta,
#                       method='SLSQP',
#                       args=(P,),
#                       constraints=const_theta,
#                       bounds=bounds_theta,
#                       options={'disp': True, "maxiter": 100000})

# Theta = res_Theta.x


# print("+++++++++++++++++++++++++++++++++++++++")
# print("after optimization")
# print("meilleurs valeurs de p", P)
# print("meilleurs valeurs de theta", Theta)
# print("rates SINR:", calculate_data_rates_SINR(res_p.x, res_Theta.x))
# print("rates SNR:", calculate_data_rates_SNR(res_p.x, res_Theta.x))
# print("+++++++++++++++++++++++++++++++++++++++")



# =============================================================================
# SNR OPTIMIZATION
# =============================================================================

# Order of Formulas for Optimizing an SNR Case

#     Initialization:
#         Initialize bounds and initial values for variables.

#     Use BCD to iteratively update each variable:
#     xi(l)←arg⁡min⁡xi∈Xif(x1(l),…,xi−1(l),xi,xi+1(l−1),…,xI(l−1))
#     xi(l)​←argxi​∈Xi​min​f(x1(l)​,…,xi−1(l)​,xi​,xi+1(l−1)​,…,xI(l−1)​)

#     Apply SCA to refine the solution by solving convex approximations:
#     x~(l+1)=arg⁡min⁡x~∈Xf~(x~;x~(l))
#     x~(l+1)=argx~∈Xmin​f~​(x~;x~(l))

#     Use FP if the objective is a ratio of two functions:
#     x(l+1)=arg⁡min⁡x∈Xu(x,y(l))v(x,y(l))
#     x(l+1)=argx∈Xmin​v(x,y(l))u(x,y(l))​

#     If constraints are present, use the penalty-based method:
#     min⁡x∈Xf(x)+ρ∑imax⁡(0,gi(x))
#     x∈Xmin​f(x)+ρi∑​max(0,gi​(x))

# By following these steps, you can optimize an SNR case effectively using the described methods.


# print("++++++++++++++++++++++++++++++++++++++")
# print("before optimization")
# print("P values before optimization ", P)
# print("Theta values before optimization ", Theta)
# print("rates SINR:", calculate_data_rates_SNR(P, Theta))
# print("+++++++++++++++++++++++++++++++++++++++")


# # Définir les contraintes de débit de données
# r_th = 100  # rate threshold
# min_p_max = 1e-4  # Minimum value for p_max
# max_p_max = 1e-2  # Maximum value for p_max

# # Generate random p_max values for each user
# p_max_values = [random.uniform(min_p_max, max_p_max) for _ in range(K)]

# # Initialize the bounds with random upper bounds
# bounds_p = Bounds([0 for _ in range(K)], p_max_values)
# # Define bounds for Theta
# bounds_theta = Bounds([0] * NR, [2 * np.pi] * NR)



# print("+++++++++++++++++++++++++++++++++++++++")
# print("after optimization")
# print("meilleurs valeurs de p", P)
# print("meilleurs valeurs de theta", Theta)
# print("rates SNR:", calculate_data_rates_SNR(res_p.x, res_Theta.x))
# print("+++++++++++++++++++++++++++++++++++++++")


# =============================================================================
# GRAPHIQUE
# =============================================================================

# Affichage des résultats après optimisation
# fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

# # Plot des valeurs de P après optimisation
# ax3.plot(range(1, K + 1), P, marker='o', linestyle='-', color='b', label='Valeurs initiales de P')
# ax3.set_xlabel('Utilisateur')
# ax3.set_ylabel('Puissance de transmission')
# ax3.set_title('Meilleures valeurs de P après optimisation')
# ax3.legend()

# # Plot des valeurs de Theta après optimisation
# ax4.plot(range(1, NR + 1), res_Theta.x, marker='o', linestyle='-', color='b', label='Meilleures valeurs de Theta')
# ax4.set_xlabel('Reflective elements')
# ax4.set_ylabel('Phase shift (radians)')
# ax4.set_title('Meilleures valeurs de Theta après optimisation')
# ax4.legend()

# plt.tight_layout()
# plt.show()

# # Calcul des débits de données après optimisation
# data_rates_opt = calculate_data_rates_SNR(res_p.x, res_Theta.x)
# print("Débits de données après optimisation:", data_rates_opt)
# =============================================================================
# =============================================================================




















# =============================================================================
# VISUALISATION GRAPHIQUE
# =============================================================================

# # Visualisation avec sympy et matplotlib
# def visualize_formula(formula, formula_str, ax):
#     ax.clear()
#     ax.text(0.5, 0.5, f"${sp.latex(formula)}$", horizontalalignment='center', verticalalignment='center', fontsize=20)
#     ax.axis('off')
#     plt.draw()

# # Création des formules sympy pour visualisation
# theta = sp.Symbol('theta')
# P_sym = sp.Symbol('P')

# SINR_sym = P_sym / (sp.Add(sp.Sum(P_sym, (P_sym, 1, K-1)), N0 * B))
# data_rate_sym = B * sp.log(1 + SINR_sym, 2)

# # Affichage des formules
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# visualize_formula(data_rate_sym, 'Data Rate Formula', ax)
# plt.show()

# # Visualisation des données après optimisation
# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.plot(range(K), calculate_data_rates_SINR(P, Theta), 'bo-', label='Data Rates')
# ax.set_xlabel('User Index')
# ax.set_ylabel('Data Rate (bps)')
# ax.set_title('Data Rates After Optimization')
# ax.legend()
# plt.grid(True)
# plt.show()
