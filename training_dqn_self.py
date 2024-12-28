import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from game_env import Game
from players import RandomPlayer, DeepQLearningAgent
import tensorflow as tf


# Entraîner un agent DQN contre lui-même
print("Training a DQN against Self")
# quelques variables globales 
tf.random.set_seed(42)
board_size = 8
buffer_size=10000
gamma = 0.99
n_actions = 64
use_target_net = True
epsilon = 0.9
version = 'v1'
batch_size = 64
supervised = False
agent_type = 'DeepQLearningAgent'

# configuration du jeu et des joueurs
p1 = DeepQLearningAgent(board_size=board_size, buffer_size=buffer_size,
                        gamma=gamma, n_actions=n_actions, 
                        use_target_net=use_target_net, epsilon=epsilon, 
                        version=version, name='dqn1')
p2 = DeepQLearningAgent(board_size=board_size, buffer_size=buffer_size,\
                        gamma=gamma, n_actions=n_actions, 
                        use_target_net=use_target_net, epsilon=epsilon,
                        version=version, name='dqn2')
p_random = RandomPlayer(board_size=board_size)
g = Game(player1=p1, player2=p2, board_size=board_size)
g2 = Game(player1=p1, player2=p_random, board_size=board_size)

# vérifier l'architecture du modèle
print("Model architecture")
p1._model.summary()

# initialisation des paramètres pour DQN
reward_type = 'current'
sample_actions = False
decay = 0.85
epsilon_end = 0.1
n_games_buffer = 300
n_games_train = 10
episodes = 1 * (10**5)
log_frequency = 500
player_update_freq = 500

# Jouer quelques parties initialement pour remplir le buffer de l'agent
start_time = time.time()
for i in tqdm(range(n_games_buffer)):
    g.reset()
    _ = g.play(add_to_buffer=True)
print("Playing %{:4d} games took %{:4d}s".format(n_games_buffer, 
                                     int(time.time() - start_time)))

model_logs = {'iteration':[], 'reward_mean':[], 'loss':[]}


# entraîner l'agent
print("Starting training against self")
for idx in tqdm(range(episodes)):
    # jouer une partie et l'ajouter au buffer
    _ = g.play(add_to_buffer=True)

    # entraînement
    loss = p1.train_agent(batch_size=batch_size)
    _ = p2.train_agent(batch_size=batch_size)

    # sélectionner le meilleur joueur
    if(idx % player_update_freq == 0):
        win_1 = 0
        # jouer 10 parties et vérifier combien de fois p1 gagne
        for j in range(20):
            winner = g.play()
            coins = g.get_players_coin()
            if(winner != -1 and coins[winner].name == "dqn1"):
                win_1 += 1;
        #  sélectionner le meilleur entre p1 et p2
        if(win_1 == 5):
            pass
        elif(win_1 > 5):
            p2.copy_weights_from_agent(p1)
        else:
            p1.copy_weights_from_agent(p2)


    # décroissance d'epsilon et réseau cible
    # sauvegarde des buffers et des modèles
    if idx % log_frequency == 0:
        model_logs['iteration'].append(idx+1)
        # jouer des parties contre un joueur aléatoire pour l'évaluation
        win_1 = 0
        for j in range(20):
            winner = g2.play()
            coins = g2.get_players_coin()
            if(winner != -1 and coins[winner].name == "dqn1"):
                win_1 += 1;
        model_logs['reward_mean'].append(round(win_1/20.0, 2))
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean','loss']]\
          .to_csv('model_logs/{:s}.csv'.format(version), index=False)
        
        # mettre à jour les réseaux cibles
        p1.update_target_net()
        p2.update_target_net()
        
        # sauvegarder les modèles
        p1.save_model(file_path='models/{:s}'.format(version), 
                      iteration=(int(idx / (n_games_train))))
        # garder un epsilon minimal pour l'entraînement
        p1.epsilon = max(p1.epsilon * decay, epsilon_end)
        # Entraîner un agent DQN contre lui-même
print("Entraînement d'un DQN contre lui-même")

# Configuration du jeu et des joueurs
p1 = DeepQLearningAgent(board_size=board_size, buffer_size=buffer_size,
                        gamma=gamma, n_actions=n_actions, 
                        use_target_net=use_target_net, epsilon=epsilon, 
                        version=version, name='dqn1')
p2 = DeepQLearningAgent(board_size=board_size, buffer_size=buffer_size,
                        gamma=gamma, n_actions=n_actions, 
                        use_target_net=use_target_net, epsilon=epsilon,
                        version=version, name='dqn2')
g = Game(player1=p1, player2=p2, board_size=board_size)

# Jouer quelques parties initialement pour remplir le buffer de l'agent
for i in tqdm(range(n_games_buffer)):
    g.reset()
    _ = g.play(add_to_buffer=True)

# Entraîner l'agent
for idx in tqdm(range(episodes)):
    g.reset()
    _ = g.play(add_to_buffer=True)
    loss = p1.train_agent(batch_size=batch_size)
    _ = p2.train_agent(batch_size=batch_size)

    # Sélectionner le meilleur joueur
    if idx % player_update_freq == 0:
        win_1 = 0
        for j in range(20):
            winner = g.play()
            coins = g.get_players_coin()
            if winner != -1 and coins[winner].name == "dqn1":
                win_1 += 1
        if win_1 > 5:
            p2.copy_weights_from_agent(p1)
        else:
            p1.copy_weights_from_agent(p2)

    # Décroissance d'epsilon et mise à jour du réseau cible
    if idx % log_frequency == 0:
        p1.update_target_net()
        p1.save_model(file_path='models/{:s}'.format(version), iteration=(int(idx / n_games_train)))
        p1.save_buffer(file_path='buffer_files/{:s}'.format(version), iteration=(int(idx / n_games_train)))
        p1.epsilon = max(p1.epsilon * decay, epsilon_end)

        # Sauvegarder les coups et les scores
        g.save_moves(file_path='moves/moves_{:04d}.txt'.format(idx))
        g.save_score(file_path='scores/score_{:04d}.txt'.format(idx))        # Entraîner un agent DQN contre lui-même
        print("Entraînement d'un DQN contre lui-même")
        
        # Configuration du jeu et des joueurs
        p1 = DeepQLearningAgent(board_size=board_size, buffer_size=buffer_size,
                                gamma=gamma, n_actions=n_actions, 
                                use_target_net=use_target_net, epsilon=epsilon, 
                                version=version, name='dqn1')
        p2 = DeepQLearningAgent(board_size=board_size, buffer_size=buffer_size,
                                gamma=gamma, n_actions=n_actions, 
                                use_target_net=use_target_net, epsilon=epsilon,
                                version=version, name='dqn2')
        g = Game(player1=p1, player2=p2, board_size=board_size)
        
        # Jouer quelques parties initialement pour remplir le buffer de l'agent
        for i in tqdm(range(n_games_buffer)):
            g.reset()
            _ = g.play(add_to_buffer=True)
        
        # Entraîner l'agent
        for idx in tqdm(range(episodes)):
            g.reset()
            _ = g.play(add_to_buffer=True)
            loss = p1.train_agent(batch_size=batch_size)
            _ = p2.train_agent(batch_size=batch_size)
        
            # Sélectionner le meilleur joueur
            if idx % player_update_freq == 0:
                win_1 = 0
                for j in range(20):
                    winner = g.play()
                    coins = g.get_players_coin()
                    if winner != -1 and coins[winner].name == "dqn1":
                        win_1 += 1
                if win_1 > 5:
                    p2.copy_weights_from_agent(p1)
                else:
                    p1.copy_weights_from_agent(p2)
        
            # Décroissance d'epsilon et mise à jour du réseau cible
            if idx % log_frequency == 0:
                p1.update_target_net()
                p1.save_model(file_path='models/{:s}'.format(version), iteration=(int(idx / n_games_train)))
                p1.save_buffer(file_path='buffer_files/{:s}'.format(version), iteration=(int(idx / n_games_train)))
                p1.epsilon = max(p1.epsilon * decay, epsilon_end)
        
                # Sauvegarder les coups et les scores
                moves_file_path = 'moves/moves_{:04d}.txt'.format(idx)
                scores_file_path = 'scores/score_{:04d}.txt'.format(idx)
                g.save_moves(file_path=moves_file_path)
                g.save_score(file_path=scores_file_path)
                print(f"Coups sauvegardés dans {moves_file_path}")
                print(f"Scores sauvegardés dans {scores_file_path}")
