import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time
from game_env import StateEnvBitBoard, Game
from players import RandomPlayer, DeepQLearningAgent
import tensorflow as tf
import matplotlib.pyplot as plt


# Train a DQN agent against a random player
if (1):
    print("Training a DQN against a random player")
    # some global variables
    tf.random.set_seed(42)
    board_size = 8
    buffer_size=10000
    gamma = 0.99
    n_actions = 64
    use_target_net = True
    epsilon = 0.9
    version = 'v1'
    batch_size = 512
    supervised = False
    agent_type = 'DeepQLearningAgent'
    
    # setup the game and players
    p1 = DeepQLearningAgent(board_size=board_size, buffer_size=buffer_size,\
                            gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,\
                            epsilon=epsilon, version=version)
    p2 = RandomPlayer(board_size=board_size)
    g = Game(player1=p1, player2=p2, board_size=board_size)
    
    # check the model architecture
    print("Model architecture")
    p1._model.summary()
    
    # initializing parameters for DQN
    reward_type = 'current'
    sample_actions = False
    decay = 0.85
    epsilon_end = 0.1
    n_games_buffer = 300
    n_games_train = 30
    episodes = 30 * (10**4)
    log_frequency = 30 * (500)
    win_list_random = []
    win_list_dqn = []
    loss = []
    
    # Play some games initially to the agent's buffer
    time_list = np.zeros(n_games_buffer)
    for i in tqdm(range(n_games_buffer)):
        start_time = time.time()
        g.reset()
        winner = g.play(add_to_buffer=True)
        if winner != -1:
            win_list_random.append(g._p[winner].name)
        else:
            win_list_random.append('Draw')
        time_list[i] = time.time() - start_time
    print('Total time taken to play {:d} games : {:.5f}s'.format(n_games_buffer, time_list.sum()))
    print('Average time per game : {:.5f}s +- {:.5f}s'.format(np.mean(time_list), np.std(time_list)))
    
    # check win %age before any training
    print("Win percentage of agent without any training")
    print(pd.Series(win_list_random).value_counts())
    
    # train the agent
    print("Starting training against a random player")
    for i in tqdm(range(episodes)):
    
        g.reset()
        _ = g.play(add_to_buffer=True)
    
    #     play a few games and fill the buffer with new data before training
        if i % n_games_train == 0:
            loss.append(p1.train_agent(batch_size=batch_size))
    
    #     epsilon decay and target_net
    #     saving buffers and models 
        if i % log_frequency == 0:
            p1.update_target_net()
            p1.save_model(file_path='models/{:s}'.format(version), iteration=(int(i / (n_games_train))))
            p1.save_buffer(file_path='buffer_files/{:s}'.format(version), iteration=(int(i / n_games_train)))
            # keep some epsilon alive for training
            p1.epsilon = max(p1.epsilon * decay, epsilon_end)
    
    #  Visualize loss
    plt.plot(loss)
    plt.title('Loss')
    plt.show()

    # Check the trained model performance against a random player
    # remove exploration 
    p1.epsilon = -1

    # Play some games and track winner
    print("Play against random player post training")
    for i in tqdm(range(n_games_buffer)):
        g.reset()
        winner = g.play()
        if winner != -1:
            win_list_dqn.append(g._p[winner].name)
        else:
            win_list_dqn.append('Draw')
    
    # Check win %age of DQN
    print("Win percentage of agent post training")
    print(pd.Series(win_list_dqn).value_counts())