# -*- coding: utf-8 -*-

# Code permettant d'avoir le graphique des temps d'éxécutions des différentes méthodes d'optimisations
import matplotlib.pyplot as plt
import time
import os

def read_time(file_path):
    try:
        with open(file_path, "r") as file:
            return float(file.read().strip())
    except FileNotFoundError:
        return None

def plot_times():
    time_a = read_time("Random_Opt_time.txt")
    time_b = read_time("Whale_Opt_time.txt")
    time_c = read_time("Alternate_Opt_time.txt")
    time_d = read_time("PP_Opt_time.txt")

    if time_a is not None and time_b is not None and time_c is not None and time_d is not None:
        # Données pour le graphique
        labels = ['Code A', 'Code B', 'Code C', 'Code D' ]
        times = [time_a, time_b]

        # Création du graphique
        plt.figure(figsize=(10, 6))
        plt.bar(labels, times, color='skyblue')
        plt.xlabel('Code')
        plt.ylabel('Temps d\'exécution (secondes)')
        plt.title('Temps d\'exécution des codes A, B, C et D')
        plt.show()

# Attendre que les deux fichiers soient créés
while not (os.path.exists("Random_Opt_time.txt") and os.path.exists("Whale_Opt_time.txt")):
    time.sleep(1)

# Appeler la fonction de création du graphique
plot_times()
