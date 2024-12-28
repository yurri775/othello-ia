import numpy as np
class ReplayBufferNumpy:
    """Cette classe stocke le tampon de replay à partir duquel les données peuvent être échantillonnées pour
    entraîner le modèle pour l'apprentissage par renforcement. Un tableau Numpy est utilisé comme
    tampon dans ce cas car il est plus facile d'ajouter plusieurs étapes à la fois, et
    l'échantillonnage est également plus rapide. C'est mieux utilisé avec l'environnement de jeu
    basé sur les tableaux Numpy

    Attributs
    ----------
    _s : Tableau Numpy
        Tampon pour stocker les états actuels,
        taille_tampon * taille_plateau * taille_plateau * 3
    _next_s : Tableau Numpy
        Tampon pour stocker les états suivants,
        taille_tampon * taille_plateau * taille_plateau * 3
    _a : Tableau Numpy
        Tampon pour stocker les actions, taille_tampon * 1
    _done : Tableau Numpy
        Tampon pour stocker l'indicateur binaire de fin
        taille_tampon * 1
    _r : Tableau Numpy
        Tampon pour stocker les récompenses, taille_tampon * 1
    _next_legal_moves : Tableau Numpy
        Tampon pour stocker les coups légaux dans l'état suivant, utile
        lors du calcul du maximum des valeurs Q dans l'état suivant
    _buffer_size : int
        Taille maximale du tampon
    _current_buffer_size : int
        Taille actuelle du tampon, peut être utilisée pour voir si le tampon est plein
    _pos : int
        Position correspondant à l'endroit où le prochain lot de données doit
        être ajouté au tampon
    _n_actions : int
        Actions disponibles dans l'environnement
    """

    def __init__(self, buffer_size=10000, board_size=8, actions=64):
        """Initialise le tampon avec la taille donnée et définit également les attributs

        Paramètres
        ----------
        buffer_size : int, optionnel
            La taille du tampon
        board_size : int, optionnel
            Taille du plateau de l'environnement
        frames : int, optionnel
            Nombre de trames utilisées dans chaque état de l'environnement
        actions : int, optionnel
            Nombre d'actions disponibles dans l'environnement
        """

    def add_to_buffer(self, s, a, r, next_s, done, legal_moves):
        """Ajoute des données au tampon, plusieurs exemples peuvent être ajoutés à la fois
        
        Paramètres
        ----------
        s : Tableau Numpy
            État actuel du plateau, doit être un état unique
        a : int
            Action actuelle prise
        r : int
            Récompense obtenue en prenant l'action sur l'état
        next_s : Tableau Numpy
            État du plateau obtenu après avoir pris l'action
            doit être un état unique
        done : int
            Indicateur binaire pour la fin du jeu
        legal_moves : Tableau Numpy
            Indicateur binaire pour les coups légaux dans l'état suivant
        """

    def get_current_size(self):
        """Renvoie la taille actuelle du tampon, à ne pas confondre avec
        la taille maximale du tampon

        Retourne
        -------
        length : int
            Taille actuelle du tampon
        """

    def sample(self, size=1000, replace=False, shuffle=False):
        """Échantillonne des données du tampon et les renvoie sous une forme facilement utilisable
        les données renvoyées ont déjà été remises en forme pour une utilisation directe dans la
        routine d'entraînement

        Paramètres
        ----------
        size : int, optionnel
            Le nombre d'échantillons à renvoyer du tampon
        replace : bool, optionnel
            Si l'échantillonnage est fait avec remplacement
        shuffle : bool, optionnel
            Redondant ici car les index sont déjà mélangés

        Retourne
        -------
        s : Tableau Numpy
            La matrice d'état pour l'entrée, taille * taille_plateau * taille_plateau * 3
        a : Tableau Numpy
            Tableau des actions prises au format one-hot, taille * nombre_actions
        r : Tableau Numpy
            Tableau des récompenses, taille * 1
        next_s : Tableau Numpy
            La matrice d'état suivant pour l'entrée
            taille * taille_plateau * taille_plateau * 3
        done : Tableau Numpy
            Indicateurs binaires pour la fin du jeu, taille * 1
        legal_moves : Tableau Numpy
            Indicateurs binaires pour les coups légaux dans l'état suivant, taille * nombre_actions
        """
        size = min(size, self._current_buffer_size)
        # select random indexes indicating which examples to sample
        idx = np.random.choice(np.arange(self._current_buffer_size), \
                                    size=size, replace=replace)

        s = self._s[idx]
        a = self._a[idx]
        r = self._r[idx].reshape((-1, 1))
        next_s = self._next_s[idx]
        done = self._done[idx].reshape(-1, 1)
        next_legal_moves = self._next_legal_moves[idx]

        return s, a, r, next_s, done, next_legal_moves
