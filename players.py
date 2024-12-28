"""Ce module contient tous les joueurs qui peuvent interagir avec le plateau d'othello"""
import numpy as np
from game_env import (StateEnvBitBoard, get_set_bits_list,
                get_total_set_bits, get_random_move_from_list,
                StateEnvBitBoardC, StateConverter)
from mcts import MCTS
import ctypes
import time
import pickle
from collections import deque
import json
from replay_buffer import ReplayBufferNumpy
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Softmax
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

def huber_loss(y_true, y_pred, delta=1):
    """Implémentation Keras pour la perte de Huber
    perte = {
        0.5 * (y_true - y_pred)**2 si abs(y_true - y_pred) < delta
        delta * (abs(y_true - y_pred) - 0.5 * delta) sinon
    }
    Paramètres
    ----------
    y_true : Tensor
        Les valeurs réelles pour les données de régression
    y_pred : Tensor
        Les valeurs prédites pour les données de régression
    delta : float, optionnel
        Le seuil pour décider d'utiliser la perte quadratique ou linéaire

    Retourne
    -------
    perte : Tensor
        valeurs de perte pour tous les points
    """
    error = (y_true - y_pred)
    quad_error = 0.5*tf.math.square(error)
    lin_error = delta*(tf.math.abs(error) - 0.5*delta)
    # quadratic error, linear error
    return tf.where(tf.math.abs(error) < delta, quad_error, lin_error)

def mean_huber_loss(y_true, y_pred, delta=1):
    """Calcule la valeur moyenne de la perte de Huber

    Paramètres
    ----------
    y_true : Tensor
        Les valeurs réelles pour les données de régression
    y_pred : Tensor
        Les valeurs prédites pour les données de régression
    delta : float, optionnel
        Le seuil pour décider d'utiliser la perte quadratique ou linéaire

    Retourne
    -------
    perte : Tensor
        perte moyenne sur les points
    """
    return tf.reduce_mean(huber_loss(y_true, y_pred, delta))

class Player:
    """La classe de base pour le joueur. Tous les attributs/fonctions communs vont ici

    Attributs
    _size : int
        la taille du plateau
    _s_input_format : str
        format d'entrée dans lequel le joueur accepte l'état du plateau 
    _legal_moves_input_format : str
        format d'entrée dans lequel le joueur accepte le masque des mouvements légaux
    _move_output_format : str
        format de sortie dans lequel le joueur retourne le mouvement sélectionné
    """
    def __init__(self, board_size=8):
        """Initialiseur

        Paramètres
        ----------
        board_size : int
            taille du plateau de jeu
        """
        self._size = board_size
        # io format for state (not relevant here)
        self._s_input_format = 'bitboard'
        # io format for legal moves
        self._legal_moves_input_format = 'bitboard_single'
        # io for output (move)
        self._move_output_format = 'bitboard_single'

    def get_state_input_format(self):
        """Retourne le format d'entrée pour l'état du jeu"""
        return self._s_input_format

    def get_legal_moves_input_format(self):
        """Retourne le format d'entrée pour les mouvements légaux"""
        return self._legal_moves_input_format

    def get_move_output_format(self):
        """Retourne le format de sortie pour le mouvement sélectionné"""
        return self._move_output_format

class RandomPlayer(Player):
    """Joueur aléatoire qui sélectionne des mouvements aléatoirement
    parmi tous les mouvements légaux
    """

    def __init__(self, board_size=8):
        Player.__init__(self, board_size)
        self.name = 'Random'


    def move(self, s, legal_moves):
        """Sélectionne un mouvement aléatoirement, étant donné l'état du plateau et
        l'ensemble des mouvements légaux

        Paramètres
        ----------
        s : tuple
            contient les bitboards noir et blanc et le joueur actuel
        legal_moves : int (64 bits)
            les états légaux sont définis à 1

        Retourne
        -------
        a : int (64 bits)
            bitboard représentant la position à jouer
        """
        if(not legal_moves):
            return 0
        return 1 << get_random_move_from_list(get_set_bits_list(legal_moves))

class DeepQLearningAgent():
    """Cet agent apprend le jeu via l'apprentissage Q
    les sorties du modèle se réfèrent partout aux valeurs Q
    Cette classe peut être étendue aux classes suivantes
    PolicyGradientAgent
    AdvantageActorCriticAgent

    Attributs
    ----------
    _board_size : int
        Taille du plateau, garder supérieur à 6 pour un apprentissage utile
        doit être le même que la taille du plateau de l'environnement
    _buffer_size : int
        Taille du buffer, combien d'exemples garder en mémoire
        doit être grand pour DQN
    _n_actions : int
        Actions totales disponibles dans l'environnement, doit être le même que l'environnement
    _gamma : float
        Remise des récompenses à utiliser pour les récompenses futures, utile dans la politique
        gradient, garder < 1 pour la convergence
    _use_target_net : bool
        Si utiliser un réseau cible pour calculer les valeurs Q de l'état suivant,
        nécessaire pour stabiliser l'apprentissage DQN
    _input_shape : tuple
        Tuple pour stocker les formes d'état individuelles
    _version : str
        chaîne de version du modèle
    _model : Graph TensorFlow
        Stocke le graphe du modèle DQN
    _s_input_format : str
        format d'entrée dans lequel le joueur accepte l'état du plateau 
    _legal_moves_input_format : str
        format d'entrée dans lequel le joueur accepte le masque des mouvements légaux
    _move_output_format : str
        format de sortie dans lequel le joueur retourne le mouvement sélectionné
    name : str
        Nom de l'agent
    epsilon: float, optionnel
        Probabilité d'exploration de l'agent, nécessaire pour la convergence DQN
    """

    def __init__(self, board_size=8, buffer_size=10000,
                 gamma=0.99, n_actions=64, use_target_net=True,
                 epsilon=0.9, version='', name='DQN'):
        """initialise l'agent

        Paramètres
        ----------
        board_size : int, optionnel
            La taille du plateau de l'environnement, garder > 6
        buffer_size : int, optionnel
            Taille du buffer, garder grand pour DQN
        gamma : float, optionnel
            Facteur de remise de l'agent, garder < 1 pour la convergence
        n_actions : int, optionnel
            Nombre d'actions disponibles dans l'environnement
        use_target_net : bool, optionnel
            Si utiliser un réseau cible, nécessaire pour la convergence DQN
        epsilon: float, optionnel
            Probabilité d'exploration de l'agent, nécessaire pour la convergence DQN
        version : str, optionnel sauf pour les modèles basés sur NN
            chemin vers l'architecture du modèle json
        """
        self._board_size = board_size
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, 3)
        self._converter = StateConverter()
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._version = version
        # io format for state (not relevant here)
        self._s_input_format = 'ndarray3d'
        # io format for legal moves
        self._legal_moves_input_format = 'bitboard_single'
        # io for output (move)
        self._move_output_format = 'bitboard_single'
        self.epsilon = epsilon
        self.name = name
        self.reset_models()

    def get_state_input_format(self):
        """Retourne le format d'entrée pour l'état du jeu"""
        return self._s_input_format

    def get_legal_moves_input_format(self):
        """Retourne le format d'entrée pour les mouvements légaux"""
        return self._legal_moves_input_format

    def get_move_output_format(self):
        """Retourne le format de sortie pour le mouvement sélectionné"""
        return self._move_output_format

    def get_gamma(self):
        """Retourne la valeur gamma de l'agent

        Retourne
        -------
        _gamma : float
            Valeur gamma de l'agent
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Réinitialise le buffer actuel 
        
        Paramètres
        ----------
        buffer_size : int, optionnel
            Initialise le buffer avec buffer_size, si non fourni,
            utilise la valeur originale
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                         self._n_actions)

    def get_buffer_size(self):
        """Obtenir la taille actuelle du buffer
        
        Retourne
        -------
        taille du buffer : int
            Taille actuelle du buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, 
                      done, next_legal_moves):
        """Ajouter l'étape de jeu actuelle au buffer de relecture

        Paramètres
        ----------
        board : Numpy array
            État actuel du plateau, peut contenir plusieurs jeux
        action : Numpy array ou int
            Action qui a été prise, peut contenir des actions pour plusieurs jeux
        reward : Numpy array ou int
            Valeur(s) de récompense pour l'action actuelle sur les états actuels
        next_board : Numpy array
            État obtenu après exécution de l'action sur l'état actuel
        done : Numpy array ou int
            Indicateur binaire pour la fin du jeu
        next_legal_moves : Numpy array
            Indicateurs binaires pour les actions autorisées aux états suivants
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, next_legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Enregistrer le buffer sur le disque

        Paramètres
        ----------
        file_path : str, optionnel
            L'emplacement pour enregistrer le buffer
        iteration : int, optionnel
            Numéro d'itération pour taguer le nom du fichier, si None, l'itération est 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Charger le buffer depuis le disque
        
        Paramètres
        ----------
        file_path : str, optionnel
            Emplacement du disque pour récupérer le buffer
        iteration : int, optionnel
            Numéro d'itération à utiliser au cas où le fichier a été tagué
            avec un, 0 si l'itération est None

        Lève
        ------
        FileNotFoundError
            Si le fichier demandé n'a pas pu être localisé sur le disque
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def reset_models(self):
        """ Réinitialiser tous les modèles en créant de nouveaux graphes"""
        self._model = self._agent_model()
        if(self._use_target_net):
            self._target_net = self._agent_model()
            self.update_target_net()

    def _prepare_input(self, board):
        """Redimensionner l'entrée et normaliser
        
        Paramètres
        ----------
        board : Numpy array
            L'état du plateau à traiter

        Retourne
        -------
        board : Numpy array
            Plateau traité et normalisé
        """
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = board.astype(np.float32)
        return board.copy()

    def _get_model_outputs(self, board, model=None):
        """Obtenir les valeurs d'action du modèle DQN

        Paramètres
        ----------
        board : Numpy array
            L'état du plateau pour lequel prédire les valeurs d'action
        model : Graph TensorFlow, optionnel
            Le graphe à utiliser pour la prédiction, modèle ou réseau cible

        Retourne
        -------
        sorties du modèle : Numpy array
            Sorties prédites du modèle sur le plateau, 
            de forme board.shape[0] * nombre d'actions
        """
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        model_outputs = model.predict_on_batch(board)
        return model_outputs

    def move(self, board, legal_moves, value=None):
        """Obtenir l'action avec la valeur Q maximale
        
        Paramètres
        ----------
        board : Numpy array
            L'état du plateau sur lequel calculer la meilleure action
        value : None, optionnel
            Gardé pour la cohérence avec d'autres classes d'agents

        Retourne
        -------
        sortie : Numpy array
            Action sélectionnée en utilisant la fonction argmax
        """
        # use the agent model to make the predictions
        if np.random.random() > self.epsilon and legal_moves:
            model_outputs = self._get_model_outputs(board, self._model)[0]
            legal_moves = self._converter.convert(legal_moves, 
                      input_format='bitboard_single', output_format='ndarray')\
                        .reshape((1,-1))[0]
            return 1 << int((63 - np.argmax(np.where(legal_moves==1, 
                                         model_outputs, -np.inf))))

        else:
            if(not legal_moves):
                a = 0
            return 1 << get_random_move_from_list(get_set_bits_list(legal_moves))

    def _agent_model(self):
        """Retourne le modèle qui évalue les valeurs Q pour une entrée d'état donnée

        Retourne
        -------
        modèle : Graph TensorFlow
            Graphe du modèle DQN
        """
        # define the input layer, shape is dependent on the board size and frames
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            m = json.loads(f.read())
        
        input_board = Input((self._board_size, self._board_size, 3), name='input')
        x = input_board
        for layer in m['model']:
            l = m['model'][layer]
            if('Conv2D' in layer):
                # add convolutional layer
                x = Conv2D(**l)(x)
            if('Flatten' in layer):
                x = Flatten()(x)
            if('Dense' in layer):
                x = Dense(**l)(x)
        out = Dense(self._n_actions, activation='linear', name='action_values')(x)
        model = Model(inputs=input_board, outputs=out)
        model.compile(optimizer=Adam(0.0005), loss='mean_squared_error')
                
        """
        input_board = Input((self._board_size, self._board_size, 3,), name='input')
        x = Conv2D(16, (3,3), activation='relu', data_format='channels_last')(input_board)
        x = Conv2D(32, (3,3), activation='relu', data_format='channels_last')(x)
        x = Conv2D(64, (4,4), activation='relu', data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dense(64, activation = 'relu', name='action_prev_dense')(x)
        # this layer contains the final output values, activation is linear since
        # the loss used is huber or mse
        out = Dense(self._n_actions, activation='linear', name='action_values')(x)
        # compile the model
        model = Model(inputs=input_board, outputs=out)
        model.compile(optimizer=RMSprop(0.0005), loss=mean_huber_loss)
        # model.compile(optimizer=RMSprop(0.0005), loss='mean_squared_error')
        """

        return model

    def set_weights_trainable(self):
        """Définir les couches sélectionnées comme non entraînables et compiler le modèle"""
        for layer in self._model.layers:
            layer.trainable = False
        # the last dense layers should be trainable
        for s in ['action_prev_dense', 'action_values']:
            self._model.get_layer(s).trainable = True
        self._model.compile(optimizer = self._model.optimizer, 
                            loss = self._model.loss)


    def get_action_proba(self, board, values=None):
        """Retourne les valeurs de probabilité d'action en utilisant le modèle DQN

        Paramètres
        ----------
        board : Numpy array
            État du plateau sur lequel calculer les probabilités d'action
        values : None, optionnel
            Gardé pour la cohérence avec d'autres classes d'agents
        
        Retourne
        -------
        sorties du modèle : Numpy array
            Probabilités d'action, la forme est board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Enregistrer les modèles actuels sur le disque en utilisant la fonction
        d'enregistrement intégrée de tensorflow (enregistre au format h5)
        en enregistrant les poids au lieu du modèle car ne peut pas charger le modèle compilé
        avec tout type d'objet personnalisé (perte ou métrique)
        
        Paramètres
        ----------
        file_path : str, optionnel
            Chemin où enregistrer le fichier
        iteration : int, optionnel
            Numéro d'itération pour taguer le nom du fichier, si None, l'itération est 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if(self._use_target_net):
            self._target_net.save_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """ charger les modèles existants, si disponibles """
        """Charger les modèles depuis le disque en utilisant la fonction
        de chargement intégrée de tensorflow (modèle enregistré au format h5)
        
        Paramètres
        ----------
        file_path : str, optionnel
            Chemin où trouver le fichier
        iteration : int, optionnel
            Numéro d'itération avec lequel le fichier est tagué, si None, l'itération est 0

        Lève
        ------
        FileNotFoundError
            Le fichier n'est pas chargé s'il n'est pas trouvé et un message d'erreur est imprimé,
            cette erreur n'affecte pas le fonctionnement du programme
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if(self._use_target_net):
            self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))
        # print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        """Imprimer les modèles actuels en utilisant la méthode summary"""
        print('Training Model')
        print(self._model.summary())
        if(self._use_target_net):
            print('Target Network')
            print(self._target_net.summary())

    def convert_bitboards(self, s, a, next_s, next_legal_moves):
        """ Convertit les états, récompenses, actions, états suivants et mouvements légaux
        échantillonnés à partir du buffer de l'agent en tableaux numpy de forme désirée
        pour entraîner le modèle

        Paramètres
        ----------
        s: ndarray de forme (self._batch_size, 3)
        a: ndarray de forme (self._batch_size, 1)
        next_s: ndarray de forme (self._batch_size, 3)
        next_legal_moves: Nunpy array de forme (self._batch_size, 1)

        Retourne
        -------
        s_board: ndarray de forme (self._batch_size, self._board_size, self._board_size, 3)
        a_board: ndarray de forme (self._batch_size, self._n_actions)
        next_s: ndarray de forme (self._batch_size, self._board_size, self._board_size, 3)
        next_legal_moves: ndarray de forme (self._batch_size, self._n_actions)

        """
        n = s.shape[0]
        s_board = np.zeros((n, self._board_size, self._board_size, 3), 
                           dtype='uint8')
        next_s_board = s_board.copy()
        a_board = np.zeros((n, self._n_actions), dtype='uint8')
        next_legal_moves_board = a_board.copy()
        for i in range(n):
            s_board[i] = self._converter.convert([int(item) for item in list(s[i])],\
                                         input_format='bitboard',\
                                         output_format='ndarray3d')
            next_s_board[i] = self._converter.convert([int(item) for item in list(next_s[i])],\
                                          input_format='bitboard',\
                                          output_format='ndarray3d')
            a_board[i] = self._converter.convert(int(a[i][0]),\
                                 input_format='bitboard_single',\
                                 output_format='ndarray').reshape(-1, self._n_actions)
            next_legal_moves_board[i] = self._converter.convert(int(next_legal_moves[i][0]),\
                               input_format='bitboard_single',\
                               output_format='ndarray').reshape(-1, self._n_actions)

        return s_board, a_board, next_s_board, next_legal_moves_board


    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Entraîner le modèle en échantillonnant à partir du buffer et retourner l'erreur.
        Nous prédisons la récompense future escomptée pour toutes
        les actions avec notre modèle. La cible pour entraîner le modèle est calculée
        en deux parties :
        1) récompense escomptée = récompense actuelle +
                        (récompense maximale possible dans l'état suivant) * gamma
           la composante de la récompense suivante est calculée en utilisant les prédictions
           du réseau cible (pour la stabilité)
        2) les récompenses pour seulement l'action prise sont comparées, donc lors
           du calcul de la cible, définir la valeur cible pour toutes les autres actions
           la même que les prédictions du modèle
        
        Paramètres
        ----------
        batch_size : int, optionnel
            Le nombre d'exemples à échantillonner à partir du buffer, doit être petit
        num_games : int, optionnel
            Non utilisé ici, gardé pour la cohérence avec d'autres agents
        reward_clip : bool, optionnel
            Si clipper les récompenses en utilisant la commande numpy sign
            récompenses > 0 -> 1, récompenses <0 -> -1, récompenses == 0 restent les mêmes
            ce paramètre peut altérer le comportement appris de l'agent

        Retourne
        -------
            perte : float
            L'erreur actuelle (la métrique d'erreur est définie dans reset_models)
        """
        s, a, r, next_s, done, next_legal_moves = \
                                    self._buffer.sample(batch_size)
        # converting states, actions and moves from bitboards to numpy arrays
        s, a, next_s, next_legal_moves = \
                        self.convert_bitboards(s, a, next_s, next_legal_moves)
        if(reward_clip):
            r = np.sign(r)
        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        next_model_outputs = self._get_model_outputs(next_s, current_model)
        # our estimate of expexted future discounted reward
        discounted_reward = np.max(np.where(next_legal_moves == 1, 
                    next_model_outputs, -np.inf), axis = 1).reshape(-1,1)
        # replace nans/inf with 0
        discounted_reward[~np.isfinite(discounted_reward)] = 0
        # we discard this value in case this is terminal state
        discounted_reward = discounted_reward * (1-done)
        # add the current step reward
        discounted_reward = r + discounted_reward
        # create the target variable, only the column with 
        # action has different value
        target = self._get_model_outputs(s)
        # we bother only with the difference in reward estimate at the 
        # selected action
        target = (1-a)*target + a*discounted_reward
        # fit
        loss = self._model.train_on_batch(s, target)
        # loss = round(loss, 5)
        return loss

    def update_target_net(self):
        """Mettre à jour les poids du réseau cible, qui est gardé
        statique pour quelques itérations pour stabiliser l'autre réseau.
        Cela ne doit pas être mis à jour très fréquemment
        """
        if(self._use_target_net):
            self._target_net.set_weights(self._model.get_weights())

    def compare_weights(self):
        """Fonction utilitaire simple pour vérifier si le modèle et le réseau cible
        ont les mêmes poids ou non
        """
        for i in range(len(self._model.layers)):
            for j in range(len(self._model.layers[i].weights)):
                c = (self._model.layers[i].weights[j].numpy() == \
                     self._target_net.layers[i].weights[j].numpy()).all()
                print('Layer {:d} Weights {:d} Match : {:d}'.format(i, j, int(c)))

    def copy_weights_from_agent(self, agent_for_copy):
        """Mettre à jour les poids entre les agents concurrents qui peuvent être utilisés
        dans l'entraînement parallèle
        """
        assert isinstance(agent_for_copy, type(self)), "Agent type is required for copy"

        self._model.set_weights(agent_for_copy._model.get_weights())
        self.update_target_net()

    def copy_buffer_from_agent(self, agent_for_copy):
        """Mettre à jour le buffer pour l'entraînement parallèle"""
        assert isinstance(agent_for_copy, type(self)), "Agent type is required for copy"
        self._buffer = agent_for_copy._buffer.copy()

class MiniMaxPlayer(Player):
    """Cet agent utilise l'algorithme minimax pour décider quel mouvement jouer

    Attributs
    _depth : int
        la profondeur à laquelle explorer les états suivants pour le meilleur mouvement
    _env : StateEnvBitBoard
        instance de l'environnement pour permettre l'exploration des états suivants
    _player : int
        stocke quelle pièce le joueur doit jouer initialement
        1 pour blanc et 0 pour noir
    """

    def __init__(self, board_size=8, depth=1):
        """Initialiseur

        Paramètres
        ----------
        board_size : int
            taille du plateau de jeu
        depth : int
            la profondeur à laquelle regarder pour les meilleurs mouvements, >= 1
        """
        self._size = board_size
        # io format for state (not relevant here)
        self._s_input_format = 'bitboard'
        # io format for legal moves
        self._legal_moves_input_format = 'bitboard_single'
        # io for output (move)
        self._move_output_format = 'bitboard_single'
        # set the depth
        self._depth = max(depth, 0)
        # an instance of the environment
        self._env = StateEnvBitBoardC(board_size=board_size)

    def move(self, s, legal_moves, current_depth=0, get_max=1,
             alpha=-np.inf, beta=np.inf):
        """Sélectionne un mouvement aléatoirement, étant donné l'état du plateau et
        l'ensemble des mouvements légaux

        Paramètres
        ----------
        s : tuple
            contient les bitboards noir et blanc et le joueur actuel
        legal_moves : int (64 bits)
            les états légaux sont définis à 1
        current_depth : int
            suit la profondeur dans la récursion
        get_max : int
            indique s'il faut jouer en tant que joueur maximum/original, 
            utile uniquement lorsque la profondeur de récursion > 1, 1 est max et 0 est min joueur
        alpha : int
            suit le maximum parmi tous les nœuds, utile pour l'élagage
        beta : int
            suit le minimum parmi tous les nœuds, utile pour l'élagage

        Retourne
        -------
        a : int (64 bit)
            bitboard représentant la position à jouer
        """
        # max player
        if(current_depth == 0):
            self._player = self._env.get_player(s)
        # get the indices of the legal moves
        move_list = get_set_bits_list(legal_moves)
        h_list = []
        m_list = []
        for m in move_list:
            s_next, legal_moves, _, done = self._env.step(s, 1 << m)
            if(current_depth < self._depth and not done):
                h_list.append(self.move(s_next, legal_moves, 
                                        current_depth+1, 1-get_max,
                                        alpha,beta))
            else:
                h_list.append(self._board_heuristics(legal_moves,
                              get_max, s_next))
            m_list.append(m)
            # print(current_depth, h_list, m, legal_moves, s, alpha, beta)
            # adjust alpha and beta
            # print(current_depth, alpha, beta, h_list[-1], 
                  # len(move_list), m, get_max)
            if(get_max):
                alpha = max(alpha, h_list[-1])
            else:
                beta = min(beta, h_list[-1])
            if(beta <= alpha):
                break
        # return the best move
        if(current_depth == 0):
            return 1 << m_list[np.argmax(h_list)]
        if(get_max):
            return alpha
        else:
            return beta
        

    def _board_heuristics(self, legal_moves, get_max, s):
        """Obtenir un nombre représentant la qualité de l'état du plateau
        ici, nous évaluons cela en comptant combien de mouvements peuvent être joués

        Paramètres
        ----------
        legal_moves : 64 bit int
            chaque bit est un indicateur pour savoir si cette position est un mouvement valide
        get_max : int
            indicateur 1 ou 0 pour déterminer quelle sortie retourner
        s : tuple
            contient les bitboards d'état

        Retourne
        -------
        h : int
            un int indiquant la qualité du plateau pour le joueur actuel
        """
        # this function uses the difference in coins in the next state
        b,w = self._env.count_coins(s)
        player = self._env.get_player(s)
        if(player):
            return w - b
        else:
            return b - w

        # this function uses the no of moves avaialable in the next state
        # and might fail later in the game when board is highly occupied
        # if(get_max):
            # return get_total_set_bits(legal_moves)
        # return -get_total_set_bits(legal_moves)

class MiniMaxPlayerC(MiniMaxPlayer):
    """Étend le MiniMaxPlayer pour implémenter une fonction de génération de mouvements purement en C"""

    def __init__(self, board_size=8, depth=1):
        """Initialiseur

        Paramètres
        ----------
        board_size : int
            taille du plateau de jeu
        depth : int
            la profondeur à laquelle regarder pour les meilleurs mouvements, >= 1
        """
        MiniMaxPlayer.__init__(self, board_size=board_size, depth=depth)
        self._env = ctypes.CDLL(r"C:\Users\marwa\OneDrive\Desktop\othello-rl-master\minimax.dll")


    def move(self, s, legal_moves):
        """Sélectionne un mouvement aléatoirement, étant donné l'état du plateau et
        l'ensemble des mouvements légaux

        Paramètres
        ----------
        s : tuple
            contient les bitboards noir et blanc et le joueur actuel
        legal_moves : int (64 bits)
            les états légaux sont définis à 1
        """
        """for C, alpha and beta cannot be passed as inf as it requires more
        dependencies to be installed, hence we will pass the max and min
        value based on our judgement
        for a heuristic based on number of moves, alpha, beta can be -+64
        for a heuristic on difference of coins
        """
        m = self._env.move(ctypes.c_ulonglong(s[0]), ctypes.c_ulonglong(s[1]), 
              ctypes.c_ulonglong(legal_moves), ctypes.c_uint(0),
              ctypes.c_uint(1), ctypes.c_int(-64), ctypes.c_int(+64), 
              ctypes.c_uint(s[2]), ctypes.c_uint(self._depth), 
              ctypes.c_uint(s[2]))
        return 1 << m

class MCTSPlayer(Player):
    """Cet agent utilise MCTS pour décider quel mouvement jouer"""

    def move(self, s, legal_moves):
        """Sélectionne un mouvement aléatoirement, étant donné l'état du plateau et
        l'ensemble des mouvements légaux

        Paramètres
        ----------
        s : tuple
            contient les bitboards noir et blanc et le joueur actuel
        legal_moves : int (64 bits)
            les états légaux sont définis à 1

        Retourne
        -------
        a : int (64 bits)
            bitboard représentant la position à jouer
        """
        # initialize mcts and train it
        mcts = MCTS(s, legal_moves, board_size=self._size)
        mcts.train()
        return mcts.select_move()

class MCTSPlayerC(Player):
    """Cet agent utilise l'implémentation MCTS en C pour décider quel mouvement jouer"""
    def __init__(self, board_size=8, n_sim=100):
        Player.__init__(self, board_size=board_size)
        self._env = ctypes.CDLL('C:/Users/marwa/OneDrive/Desktop/othello-rl-master/mcts.dll')
        self._n_sim = n_sim

    def move(self, s, legal_moves):
        """Sélectionne un mouvement aléatoirement, étant donné l'état du plateau et
        l'ensemble des mouvements légaux

        Paramètres
        ----------
        s : tuple
            contient les bitboards noir et blanc et le joueur actuel
        legal_moves : int (64 bits)
            les états légaux sont définis à 1

        Retourne
        -------
        m : int (64 bits)
            bitboard représentant la position à jouer
        """
        # train mcts and get the move
        m = self._env.move(ctypes.c_ulonglong(s[0]), ctypes.c_ulonglong(s[1]), 
                    ctypes.c_uint(s[2]), ctypes.c_ulonglong(legal_moves), 
                    ctypes.c_uint(self._n_sim))
        return 1 << m
