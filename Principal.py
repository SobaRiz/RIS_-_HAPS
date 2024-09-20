import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
import random
import sympy as sp
import json

# =============================================================================
# Ce code sert à contenir les fonctions principales appelés dans les autres scripts.

# MOTS CLES (définition à retrouver dans le READ.ME):
# ----------------------------------------------------
    # SNR (Signal-to-Noise Ratio)
    # SINR (Signal-to-Interference-plus-Noise Ratio)
    # Débit de données
    # Puissance de transmission
    # Puissance du bruit blanc (AWGN)
    # Nakagami-m fading
    # HAPS (High-Altitude Platform Station)
    # RIS (Reconfigurable Intelligent Surface)
    # Canaux de communication
    # Matrices de canaux (H_UR, H_RH)
    # Phase-shift
    # Antenne
    # Gain d'antenne
    # Modulation de signal
    # Distance
    # Angle
    # Bandwidth (Bande passante)
    # Fréquence porteuse (Carrier frequency)
    # Interférence
    # Phase-shift Theta
# ----------------------------------------------------
# =============================================================================

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

# Récupérer les coordonnées de HAPS
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

# Récupérer les coordonnées de RIS
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


# Récupérer les coordonnées de Users
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
# Simplifie le calcul des matrices de canaux en encapsulant la création des vecteurs de réponse d'antenne.
# Définition de la fonction pour calculer la matrice g(x, φ)
def calculate_g(x, phi):
    return ((1 / np.sqrt(x)) * np.array([np.exp(1j * l * np.pi * np.cos(phi)) for l in range(x)]))


# Formule 1
# Calcule la matrice de canal entre l'Unité Utilisateur (User, U) et le Répéteur.
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
# Calcule la matrice de canal entre le Répéteur (R) et la station haute altitude (HAPS)
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

# Voir le code de Mme BOUBAKAR Eya pour plus de spécifité sur la partie SINR

# γk = Pk||HkRH ΘkhkU R|| / BkN0, ∀k = 1, . . . , K
def calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k):
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

    return np.sum(data_rates_SNR)

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

# On cherche ici à maximiser la somme des débits de données.
# Cependant, étant donné que la plupart des algorithmes d'optimisation sont des algorithmes de minimisation, on définit ici une fonction objectif qui retourne l'opposé de ce qu'on souhaite maximiser
# L'objectif est de trouver les valeurs optimales des puissances P et des phases Θ qui maximisent la somme des débits de données dans le système de communication.

# Fonction objectif pour P SNR
def negative_sum_rate_p_fixed_SNR(system_model, P, Theta, N_R_k, N_H_k):
    data_rates_SNR = calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k)
    return - np.sum(data_rates_SNR)

# Fonction de coût pour l'optimisation
def negative_sum_rate_theta_fixed_SNR(Theta, P, system_model, N_R_k, N_H_k):
    data_rates = calculate_data_rates_SNR(system_model, P, Theta, N_R_k, N_H_k)
    return -np.sum(data_rates)
