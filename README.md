# RIS_-_HAPS
Will have some modifications later

# =============================================================================
# Liens et explications
# =============================================================================

Liste des fichiers :
  json_datas.json :
    [Opérationnel]
    Ce fichier contient toutes les constantes redondantes au sein des codes suivants. Il est l'unique fichier qui sera modifié lors des tests.
  Principal.py :
    [Opérationnel]
    Ce code contient les fonctions principales appelés dans les autres scripts.
  Random_Opt.py : 
    [Opérationnel]
    Ce code met en place la méthode d'optimisation Random avec les composants SNR présent dans le fichier "Principal.py".
  Whales_Opt.py :
    [Semi-Opérationnel]
    Ce code met en place la méthode d'optimisation des baleines avec les composants SNR présent dans le fichier "Principal.py".
  TIME.py :
    [Absolument pas Opérationnel]
    Ce code permet de regrouper les valeurs des temps d'éxécutions des différents codes dans un seul graphique.

Lien entre les fichiers :
json_datas -> n'appel personne
Principal -> appel json
Random -> appel json, principal
Whales -> appel json, principal
Time -> appel random, whales

  |----------> Principal <---------|
  ↑               ↓                ↑
Random --------> Json <-------- Whales
  ↑                               ↑
  |------------- Time ------------|

# =============================================================================
# Définition des mots clés
# =============================================================================

**Antenne** : Un dispositif utilisé pour transmettre ou recevoir des signaux électromagnétiques.

**Bandwidth (Bande passante)** : La gamme de fréquences qu'un canal de communication peut transmettre, influençant la quantité de données pouvant être envoyées simultanément.

**Canaux de communication** : Les voies par lesquelles les signaux sont transmis d'un émetteur à un récepteur.

**Débits de données**: La quantité de données qui peut être transmise sur un canal de communication par unité de temps.

**Fréquence porteuse (Carrier frequency)** : La fréquence d'un signal porteur sur lequel des informations sont superposées pour la transmission.

**Gain d'antenne** : Une mesure de l'efficacité d'une antenne à diriger l'énergie du signal dans une direction particulière.

**HAPS (High Altitude Platform Station)** : Des stations situées à haute altitude qui fournissent des services de communication et de surveillance à grande échelle.
  *Création* : Développée par des chercheurs, notamment à partir des années 2000, pour améliorer les communications sans fil depuis des plateformes en altitude.
  *Pourquoi* : Utilisée pour fournir une couverture réseau dans des zones difficiles d'accès et pour des applications spécifiques.
  *Fonctionnement* : Station de plateforme à haute altitude (HAPS) fonctionne comme un relais pour les signaux de communication.
  *Utilisation* : Appliquée dans les télécommunications, l'Internet des objets (IoT) et la surveillance environnementale.
  *Comment ça fonctionne* :
      - Positionnement élevé : Place des équipements de communication à haute altitude pour une meilleure portée.
      - Transmission de signaux : Relaye des signaux entre les utilisateurs au sol et les satellites ou réseaux terrestres.
      - Couverture flexible : Adapte la couverture selon la demande et les conditions environnantes.

**Interférence** : La perturbation d'un signal causée par d'autres signaux, affectant la qualité de la communication.

**Meilleures valeurs de P (puissances optimisées)** : Les niveaux de puissance optimaux calculés pour chaque utilisateur afin d'améliorer le débit de données et la performance globale.

**Matrices de canaux (H_UR, H_RH)** : Représentations mathématiques des canaux de communication entre utilisateurs et stations, utilisées dans l'analyse des performances.

**Nakagami-m fading**: Un modèle statistique utilisé pour décrire la variation de la puissance du signal due aux changements de l'environnement radio.
  *Création* : Introduit par Nakagami en 1960 pour modéliser les variations de puissance du signal dans des environnements de communication.
  *Pourquoi* : Utilisé pour analyser les performances des systèmes de communication dans des conditions de propagation complexes.
  *Fonctionnement* : Modélise la dégradation du signal causée par des facteurs environnementaux comme l'atténuation et la réflexion.
  *Utilisation* : Appliqué dans les études de communication sans fil, la conception de réseaux et l'optimisation de systèmes de transmission.
  *Comment ça fonctionne* :
      - Modèle statistique : Utilise une distribution de Nakagami pour décrire la puissance du signal reçu.
      - Paramètre m : Le paramètre m contrôle la sévérité de l'atténuation, affectant la forme de la distribution.
      - Analyse des performances : Évalue les performances des systèmes en prenant en compte les variations de signal sur des distances spécifiques.

**NR (nombre d'éléments réfléchissants dans le RIS)** : Le nombre total d'éléments dans une surface intelligente reconfigurable, influençant la capacité du système à rediriger les signaux.

**P_max (puissance maximale de transmission pour chaque utilisateur)** : La limite supérieure de puissance qu'un utilisateur peut émettre pour éviter les interférences et optimiser le signal.

**Phase shift (décalage de phase en radians)** : La différence de phase entre deux signaux, mesurée en radians, affectant l'interférence constructive ou destructive dans les communications.

**Puissance du bruit blanc (AWGN)** : Une mesure du bruit de fond dans un canal de communication, considéré comme ayant une puissance constante sur toutes les fréquences et ne pouvant disparaitre.
  *Création* : Modèle théorique établi dans le domaine des communications au milieu du 20e siècle.
  *Pourquoi* : Utilisé pour évaluer et concevoir des systèmes de communication en tenant compte des interférences dues au bruit.
  *Fonctionnement* : Représente le bruit de fond dans les systèmes de communication, qui est considéré comme un ajout aléatoire au signal.
  *Utilisation* : Appliqué dans l'analyse des performances des systèmes de communication, les tests de transmission, et les simulations.
  *Comment ça fonctionne* :
      - Caractéristiques : Le bruit est dit "additif" car il s'ajoute au signal, "blanc" car il a une densité spectrale constante, et "gaussien" car il suit une distribution normale.
      - Modélisation : Modélisé par un processus aléatoire où les valeurs du bruit à différents moments sont indépendantes et identiquement distribuées.
      - Impact sur les systèmes : Affecte le rapport signal sur bruit (SNR), influençant la capacité de transmission et la qualité du signal.

**Random optimization (algorithme d'optimisation aléatoire)** : Une méthode d'optimisation qui génère des solutions aléatoires pour explorer l'espace de recherche efficacement.
  *Création* : Développée en tant qu'algorithme d'optimisation général.
  *Pourquoi* : Utilisée pour explorer des espaces de solution complexes de manière simple et efficace.
  *Fonctionnement* : Génère des solutions aléatoires et les évalue pour trouver la meilleure.
  *Utilisation* : Appliquée dans divers domaines tels que l'ingénierie, l'IA, et la recherche opérationnelle.
  *Comment ça fonctionne* :
      - Génération aléatoire : Crée des solutions initiales dans l'espace de recherche.
      - Évaluation : Mesure la qualité de chaque solution.
      - Sélection : Identifie la meilleure solution pour l'itération suivante.

**RIS (Reconfigurable Intelligent Surfaces)** : Des surfaces capables de modifier la propagation des signaux radio en ajustant dynamiquement leurs propriétés.
  *Création* : Développée dans les recherches sur les technologies de communication au cours des années 2010.
  *Pourquoi* : Utilisée pour améliorer la performance des réseaux sans fil en manipulant la propagation des ondes radio.
  *Fonctionnement* : Permet de contrôler la réflexion des signaux radio grâce à des surfaces reconfigurables dotées de nombreux éléments actifs.
  *Utilisation* : Appliquée dans les réseaux 5G, la communication sans fil, et l'optimisation de la couverture réseau.
  *Comment ça fonctionne* :
      - Structure : Composée d'éléments réfléchissants qui peuvent être ajustés individuellement pour optimiser la direction et la phase des signaux.
      - Contrôle : Utilise des techniques d'intelligence artificielle pour adapter les paramètres de la surface en temps réel selon les conditions du réseau.
      - Avantages : Améliore la capacité, la couverture et la qualité du signal, tout en réduisant les interférences.

**SNR (Signal-to-Noise Ratio)** : Un rapport mesurant la puissance du signal par rapport au bruit de fond, essentiel pour évaluer la qualité de la communication.
  *Création* : Concept établi dans les domaines de l'ingénierie et des communications depuis le début du XXe siècle.
  *Pourquoi* : Utilisé pour évaluer la qualité d'un signal dans un environnement bruité.
  *Fonctionnement* : Mesure le rapport entre la puissance d'un signal utile et la puissance du bruit de fond.
  *Utilisation* : Appliqué dans les systèmes de communication, le traitement du signal, et l'audiovisuel.
  *Comment ça fonctionne* :
      - Calcul : SNR est généralement exprimé en décibels (dB) : 
      SNR = [10(log)_⁡10](Psignal/Pbruit).
      - Interprétation : Un SNR plus élevé indique une meilleure qualité de signal, tandis qu'un SNR faible peut mener à des erreurs de transmission.
      - Impact : Crucial pour la performance des systèmes sans fil et des réseaux de communication.

**SINR (Signal-to-Interference-plus-Noise Ratio)** : Un rapport mesurant la puissance d'un signal par rapport à la somme des interférences et du bruit, essentiel pour la qualité de la réception.
  *Création* : Concept développé dans le cadre des communications sans fil, principalement depuis les années 1990.
  *Pourquoi* : Utilisé pour évaluer la qualité d'un signal dans des environnements avec interférence et bruit.
  *Fonctionnement* : Mesure le rapport entre la puissance d'un signal utile et la somme de la puissance de l'interférence et du bruit.
  *Utilisation* : Appliqué dans les systèmes de communication sans fil, les réseaux cellulaires, et le traitement du signal.
  *Comment ça fonctionne* :
      - Calcul : SINR est souvent exprimé en décibels (dB) :
      SINR=(10.log⁡_10)[Psignal/(Pinterférence+Pbruit)]
      - Interprétation : Un SINR élevé indique une meilleure qualité de communication, tandis qu'un SINR faible peut entraîner des dégradations de performance.
      - Impact : Essentiel pour optimiser les performances des réseaux sans fil, notamment dans la gestion des ressources radio.

**Theta (phases ajustables)** : Des valeurs représentant les décalages de phase des signaux dans les systèmes de communication, influençant la propagation des ondes.

**Population de baleines** : Un ensemble de solutions dans l'algorithme d'optimisation par baleines, représentant différentes configurations possibles.
  *Création* : Proposé par Seyedali Mirjalili en 2016.
  *Pourquoi* : Inspiré par la chasse des baleines à bosse pour résoudre des problèmes complexes d'optimisation.
  *Fonctionnement* : Simule l'encerclement et l'attaque en spirale pour trouver la meilleure solution.
  *Utilisation* : Applications dans l'IA, traitement de signal, ingénierie, optimisation de réseaux, planification, et clustering.
  *Comment ça fonctionne* : L'optimisation suit trois phases :
      - Encerclement de la proie : Ajuste la position pour se rapprocher de la meilleure solution.
      - Attaque en spirale : Approche la solution en spirale.
      - Recherche de proie : Exploration aléatoire pour éviter les optima locaux.
