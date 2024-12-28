"""Cette classe contient la structure de l'arbre et les algorithmes pertinents pour
effectuer une itération de la recherche d'arbre de Monte Carlo"""
import numpy as np
from game_env import StateEnvBitBoard, get_total_set_bits, \
                get_set_bits_list, get_random_move_from_list, StateEnvBitBoardC
from collections import deque

class Node:
    """MCTS est un arbre composé de plusieurs nœuds qui stockent des informations statistiques
    sur les simulations effectuées jusqu'à présent, et sont utilisés pour calculer
    les meilleurs coups

    Attributs
    ----------
    s : tuple
        tuple contenant les états du plateau sous forme de bitboards
    legal_moves : int 64 bits
        les bits correspondant aux positions légales sont mis à 1
    w : int
        le nombre de victoires lorsque ce nœud faisait partie du déploiement
    n : int
        le nombre de simulations lorsque ce nœud faisait partie du déploiement
    N : int
        le nombre de simulations où le nœud parent de ce nœud
        faisait partie du déploiement
    children : list
        liste contenant les nœuds enfants de ce nœud
    m : int
        le coup joué pour arriver à cet état
    terminal : int
        indicateur pour savoir si ce nœud est terminal dans l'arbre, identique à done
    parent : int
        l'index du parent de ce nœud, dans la liste des nœuds mcts
    """
    def __init__(self, s, legal_moves, m=-1, terminal=0, parent=None):
        """Initialiseur pour la classe node, l'arbre est une collection de nœuds

        Paramètres
        ----------
        s : tuple
            l'état du plateau représenté sous forme de bitboards
        legal_moves : int 64 bits
            les bits correspondant aux positions légales sont mis à 1 dans cet int
        m : int (optionnel)
            le coup joué pour arriver à cet état, cela désigne la position
            depuis l'extrémité droite du tableau (pas 64 bits)
        terminal : int (optionnel)
            indicateur indiquant si c'est un nœud feuille
        parent : int (optionnel)
            index du parent de ce nœud dans la liste des nœuds mcts
        """
        self.state = s
        self.legal_moves = legal_moves
        # convertir les coups légaux de 64 bits en un ensemble de positions
        # pour une utilisation rapide plus tard
        self.legal_moves_set = get_set_bits_list(legal_moves)
        np.random.shuffle(self.legal_moves_set)
        # pour comparer si tous les enfants ont été ajoutés ou non
        self.total_legal_moves = get_total_set_bits(legal_moves)
        self.w = 0
        self.n = 0
        self.N = 0
        self.children = []
        # puisque nous avons mélangé le legal_moves_set, nous pouvons utiliser total_children
        # comme l'index à partir duquel nous devons choisir le prochain coup inexploré
        self.total_children = 0
        self.move = m
        self.terminal = terminal
        self.parent = parent

    def add_child(self, idx):
        """Ajouter le nœud donné comme enfant au nœud actuel

        Paramètres
        ----------
        node : idx
            index dans la liste des nœuds mcts
        """
        self.children.append(idx)
        self.total_children += 1

    def get_ucb1(self, c=np.sqrt(2)):
        """Obtenir la borne supérieure de confiance sur le nœud actuel
        ucb1 = (w/n) + c*sqrt(ln(N)/n)

        Retours
        -------
        ucb1 : float
        """
        return ((0.1*self.w)/self.n) + c * np.sqrt(np.log(self.N)/self.n)


class MCTS:
    """Classe MCTS contenant également la fonction d'entraînement et d'inférence
    
    Attributs
    ----------
    _c : float
        coefficient d'exploration
    _node_list : list
        liste des nœuds dans l'arbre mcts, la racine est à l'index 0
        l'attribut children stockera une liste d'indices correspondant
        à cet arbre (liste de nœuds), l'attribut parent d'un nœud contiendra également un index
        se référant à cet arbre; cela a été fait car il n'y a pas de pointeurs en python
        et cela aide à éviter les répétitions de nœuds dans la liste des enfants, etc.
    _env : StateEnvBitBoard
        instance de l'environnement d'état
    """
    def __init__(self, s, legal_moves, board_size=8, c=np.sqrt(2)):
        """Initialiseur pour la classe MCTS et initialise sa racine

        Paramètres
        ----------
        s : tuple
            l'état du plateau représenté sous forme de bitboards
        legal_moves : int 64 bits
            les bits correspondant aux coups légaux sont mis à 1
        c : float (optionnel)
            paramètre pour l'exploration dans UCB, par défaut sqrt(2)
        """
        self._c = c
        self._node_list = [Node(s.copy(), legal_moves)]
        self._env = StateEnvBitBoardC(board_size)

    def get_not_added_move(self, node):
        """Sélectionner aléatoirement un coup parmi ceux qui n'ont pas encore été joués

        Paramètres
        ----------
        node : Node
            le nœud pour lequel sélectionner un coup non ajouté à l'arbre

        Retours
        -------
        m : int
            la position (indexation depuis l'extrémité droite) où jouer le coup
            pour utiliser dans l'environnement bitboard, passer 1<<m au lieu de m
        """
        all_moves = node.legal_moves_set.copy()
        for c in node.children:
            all_moves.remove(self._node_list[c].move)
        all_moves = list(all_moves)
        return get_random_move_from_list(all_moves)

    def train(self, n=100):
        """Entraîner l'arbre MCTS pour un nombre n d'itérations

        Paramètres
        ----------
        n : int (optionnel)
            le nombre d'étapes de simulation à exécuter
        """
        while(n):
            n -= 1
            ##############################
            ####### Phase de Sélection ######
            ##############################
            """Sélectionner un nœud dans l'arbre qui n'est ni un nœud feuille
            ni entièrement exploré"""
            e = 0
            while(True):
                node = self._node_list[e]
                if(node.total_legal_moves != \
                   node.total_children or \
                   node.terminal == 1):
                    # au moins un coup inexploré est présent, arrêter la
                    # sélection ici
                    break
                else:
                    # puisque tous les nœuds du nœud précédent ont été explorés au moins
                    # une fois, nous passons au niveau suivant et sélectionnons l'enfant 
                    # avec le plus haut ucb1
                    next_node = None
                    best_ucb1 = -np.inf
                    for idx in node.children:
                        ucb1 = self._node_list[idx].get_ucb1(self._c)
                        if(ucb1 > best_ucb1):
                            best_ucb1 = ucb1
                            next_node = idx
                    e = next_node
            # cela par défaut à la racine au cas où la condition else n'est pas exécutée
            node, node_idx = self._node_list[e], e
            
            ##############################
            ####### Phase d'Expansion ######
            ##############################
            """Sélectionner un des nœuds enfants pour ce nœud qui est 
            inexploré"""
            if(not node.terminal):
                """d'abord obtenir un coup aléatoire parmi les coups qui n'ont pas 
                été ajoutés à l'arbre mcts"""
                # m = self.get_not_added_move(node)
                m = node.legal_moves_set[node.total_children]
                # jouer le jeu et ajouter un nouveau nœud à l'arbre (liste de nœuds)
                next_state, next_legal_moves, _, done = \
                                    self._env.step(node.state, 1<<m)
                node = Node(s=next_state.copy(), legal_moves=next_legal_moves, 
                            m=m, terminal=done, parent=e)
                # ajouter le nœud à la liste des nœuds
                self._node_list.append(node)
                # ajouter l'index dans cette liste à la liste des enfants du parent
                self._node_list[e].add_child(len(self._node_list)-1)
                node_idx = len(self._node_list)-1

            ##############################
            ###### Phase de Simulation ######
            ##############################
            """Jouer jusqu'à la fin en sélectionnant aléatoirement des coups à partir du
            nœud nouvellement créé (dans le cas d'un nœud terminal, cette étape est ignorée"""
            s = node.state
            legal_moves = node.legal_moves
            if(node.terminal != 1):
                done = 0
                while(not done):
                    a = get_random_move_from_list(get_set_bits_list(legal_moves))
                    s, legal_moves, _, done = self._env.step(s, 1<<a)
            winner = self._env.get_winner(s)

            ##############################
            #### Phase de Rétropropagation ###
            ##############################
            """Rétropropager la valeur du gagnant du nœud (à partir duquel nous avons commencé
            à jouer) à la racine pour mettre à jour les paramètres statistiques de chaque nœud"""
            while(True):
                node.n += 1
                # mettre à jour la valeur de N dans les enfants
                for c in node.children:
                    self._node_list[c].N = node.n
                if(winner != -1):
                    node.w += (1-winner == self._env.get_player(node.state))
                else:
                    # égalité
                    node.w += 0.5
                # monter d'un niveau
                if(node.parent is None):
                    break
                else:
                    node, node_idx = self._node_list[node.parent], node.parent

    def select_move(self):
        """Sélectionner le meilleur coup après que l'arbre a été entraîné
        ici nous sélectionnons celui avec le plus grand nombre de jeux

        Retours
        -------
        m : int 64 bits
            int le drapeau correspondant au meilleur coup mis à 1
        """
        most_plays = -np.inf
        m = -1
        for c in self._node_list[0].children:
            node = self._node_list[c]
            if(node.n > most_plays):
                m = node.move
                most_plays = node.n
        return 1 << m