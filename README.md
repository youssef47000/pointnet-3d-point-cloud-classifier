# Classification de Formes Géométriques 3D 

Ce projet propose une implémentation de l'architecture **PointNet** pour classer des nuages de points 3D générés de manière synthétique. L'objectif est de reconnaître la structure d'objets composés de 2048 points, en restant insensible à leur orientation ou à l'ordre des points.

## Les 3 Classes d'Objets

Le jeu de données est généré par le script `prepare_data.py` et se compose de trois catégories distinctes :
* **Classe 00 : Cylindres**.
* **Classe 01 : Parallélépipèdes**.
* **Classe 02 : Tores (Donuts)**.

## Fonctionnement du Modèle

Le réseau s'appuie sur les deux piliers de PointNet pour traiter les données géométriques :
* **T-Net (Transformation Network)** : Le modèle prédit une matrice d'alignement pour "redresser" l'objet avant de l'analyser, ce qui permet de gérer les rotations aléatoires appliquées aux données.
* **Signature Globale (Max Pooling)** : Le réseau identifie les points critiques de la forme pour créer un descripteur global invariant à l'ordre des points.

## Robustesse au Bruit

Le modèle intègre une stratégie d'augmentation de données pour améliorer sa généralisation :
* **Bruit Gaussien** : Un bruit aléatoire est injecté sur les coordonnées des points pendant l'entraînement.
* **Résultat** : Avec un niveau de bruit (sigma) de 0.10, le modèle maintient une précision d'environ **90.55 %** lors des tests.

## Utilisation 

### Entraînement et évaluation

```bash
python tp_ptnet_skel.py