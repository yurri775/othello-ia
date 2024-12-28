from game_env import (StateEnv, Game, StateEnvBitBoard, StateConverter, 
                            StateEnvBitBoardC)
from players import RandomPlayer
import numpy as np
import time
from tqdm import tqdm

board_size = 8

# initialize classes
p1 = RandomPlayer(board_size=board_size)
p2 = RandomPlayer(board_size=board_size)
g = Game(player1=p1, player2=p2, board_size=board_size)
# chemin: /C:/Users/marwa/OneDrive/Desktop/othello-rl-master/test_script.py

def convert_boards(s, m=None):
    """Convertit l'état du plateau de bitboard en ndarray

    Paramètres
    ----------
    s : list
        contient les bitboards noir, blanc et le joueur actuel
    m : int (64 bits), par défaut None
        bitboard pour les coups légaux
    
    Retourne
    -------
    s : list
        contient les tableaux de plateau noir, blanc et le joueur actuel
    m : ndarray
        tableau des coups légaux
    """

def compare_boards(s, correct_s, case_no):
    """Compare les plateaux de l'environnement de jeu avec les versions correctes

    Paramètres
    ----------
    s : list
        [état du plateau, coups légaux, joueur actuel]
        état du plateau = [tableau plateau noir, tableau plateau blanc, joueur actuel]
        quand la fonction step est appliquée, done doit aussi être vérifié
    correct_s : list
        même format que ci-dessus mais avec les valeurs correctes
    case_no : int
        le cas actuel évalué, utilisé pour l'affichage
        
    Retourne
    -------
    success : bool
        indique si tous les cas de test ont réussi
    """

# cas de test pour vérifier si l'environnement fonctionne correctement

######## Cas 1: reset est correct ########

######## Cas 2: jouer un coup sur le plateau initial ########

######## Cas 3: un coup qui résulte en tous sauf un noir ########

######## Cas 4: un coup qui résulte en tous sauf un blanc ########

######## Cas 5: plateau personnalisé d'un match ########
# https://www.worldothello.org/about/world-othello-championship/woc/2017/round/5
# Maria Serena Vecchi 13-51 Caroline Nicolas, coups 53 - 54

######## Cas 6: plateau personnalisé d'un match ########
# même match que ci-dessus

######## Cas 7: plateau personnalisé d'un match ########
# même match que ci-dessus

######## Cas 8: plateau personnalisé d'un match ########
# https://www.worldothello.org/about/world-othello-championship/woc/2017/round/5
# Niklas Wettergren 31-33 Brian Rose, coups 40 - 41

# vérifier le temps pris pour jouer 10000 parties

# enregistrer et sauvegarder la partie

# cas de test pour vérifier si l'augmentation du plateau est correctement implémentée
# d'abord un bloc de code pour vérifier quelles transformations sont redondantes

def get_board_augmentations(transition):
    """
    obtient la liste de toutes les transitions uniques obtenues par retournement, rotations etc
    en considérant seulement les rotations anti-horaires, les transitions sont normal, rotation normale 270
    rotation normale 180, rotation normale 90, retournement vertical, retournement vertical rotation 270,
    retournement vertical rotation 180 et retournement vertical rotation 90

    Paramètres 
    ----------
    transition : list
        contient [[bitboard noir, bitboard blanc, joueur actuel], coups légaux, joueur actuel,
                    action, [prochain bitboard noir, prochain bitboard blanc, prochain joueur],
                    prochains coups légaux, prochain joueur, terminé, gagnant]
    
    Retourne
    -------
    transition_list : list
        liste des transitions augmentées
    """

# cas de test
# vérifier le temps pris pour jouer 10000 parties avec augmentation (8 * 10000)
    transition_list = []
    fa = lambda x: conv.convert(x, input_format='bitboard_single', output_format='ndarray')
    fb = lambda x: conv.convert(x, input_format='ndarray', output_format='bitboard_single')
    for f1 in [lambda x: x, np.flipud]:
        for f2 in [lambda x: x,
                   lambda x: np.rot90(np.rot90(np.rot90(x))),
                   lambda x: np.rot90(np.rot90(x)),
                   lambda x: np.rot90(x)]:
            f_temp = lambda x: fb(f2(f1(fa(x))))
            transition_list.append([[f_temp(transition[0][0]), f_temp(transition[0][1]), transition[0][2]],
                                       f_temp(transition[1]), transition[2], f_temp(transition[3]),
                                        transition[4], transition[5]])
    return transition_list

# test cases
while(1):
    print('RUNNING TEST CASES FOR BOARD AUGMENTATION FUNCTION')
    success = True
    env = StateEnvBitBoard(board_size=board_size)
    conv = StateConverter()  
    augmentation = ['normal', 'normal rot 270', 'normal rot 180', 'normal rot 90',
                    'vertical flip', 'vertical flip rot 270', 'vertical flip rot 180', 'vertical flip rot 90']
    ######## Case 1: custom board from a match ########
    # https://www.worldothello.org/about/world-othello-championship/woc/2017/round/5
    # Maria Serena Vecchi 13-51 Caroline Nicolas, move 53 - 54
    print('Running Case 1: Custom board from a match')
    base_transition = [[9055374549248, 827328404604, 1], 8011054621671425, 1, 1<<46, \
                     0, 0]
    correct_transitions = get_board_augmentations(base_transition)
    augmented_transitions = g.create_board_reps(base_transition)
    for i in range(len(correct_transitions)):
        if(correct_transitions[i] != augmented_transitions[i]):
            success = False
            print('Case {:d} Augmentation {:s} does not match'.format(1, augmentation[i]))
            print('Expected')
            print(correct_transitions[i])
            print('Got')
            print(augmented_transitions[i])
    
    ######## Case 2: custom board from a match ########
    # https://www.worldothello.org/about/world-othello-championship/woc/2017/round/5
    # Niklas Wettergren 31-33 Brian Rose, moves 40 - 41 
    print('Running Case 2: Custom board from a match')
    base_transition = [[8847673277496, 4340405112184963328, 0], 4774871685931205120, 0, 1<<54, \
                    0, 0]
    correct_transitions = get_board_augmentations(base_transition)
    augmented_transitions = g.create_board_reps(base_transition)
    for i in range(len(correct_transitions)):
        if(correct_transitions[i] != augmented_transitions[i]):
            success = False
            print('Case {:d} Augmentation {:s} does not match'.format(1, augmentation[i]))
            print('Expected')
            print(correct_transitions[i])
            print('Got')
            print(augmented_transitions[i])
    
    if(success):
        print('Passed all test cases for board augmentation ! Congrats !')
    else:
        print('One or more test cases failed for board augmentation, correct code and try again !')
    # break from the while loop
    break

# check time taken to play 10000 games with augmentation (8 * 10000)
if(success):
    total_games = 1000
    total_reps = 8
    time_list = np.zeros(total_games)
    for i in tqdm(range(total_games)):
        start_time = time.time()
        g.reset()
        winner = g.play()
        _ = g.get_game_history(augmentations=True)
        time_list[i] = time.time() - start_time
    # print results
    print('Total time taken to play {:d} games with {:d} augmentations : {:.5f}s'.format(total_games, total_reps, time_list.sum()))
    print('Average time per game : {:.5f}s +- {:.5f}s'.format(np.mean(time_list)/total_reps, np.std(time_list)/total_reps))
