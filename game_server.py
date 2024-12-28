# python game_server.py
import os
from flask import Flask, request, render_template, jsonify, send_file
from players import RandomPlayer, MiniMaxPlayerC, MCTSPlayerC
from game_env import StateEnvBitBoardC, get_set_bits_list, StateConverter
from random import random

app = Flask(__name__)

"""une liste centralisée des joueurs IA disponibles
celle-ci est utilisée pour afficher la première page où l'utilisateur peut sélectionner
contre quelle IA jouer"""

ai_players = {
    'random': ['Random Player', RandomPlayer],
    'minimax': ['MiniMax Player (with alpha-beta pruning)', MiniMaxPlayerC],
    'mcts': ['Monte Carlo Tree Search (MCTS)', MCTSPlayerC]
}

# initialisations liées au déroulement du jeu, variables globales
ai_player = None
ai_player_coin = 0
env = StateEnvBitBoardC()
conv = StateConverter()
board_state = None
board_legal_moves = None

def ai_player_move(s, legal_moves):
    """
    La même fonction est utilisée dans la classe Game de game_env
    cette fonction gère toutes les manipulations liées à la conversion des coups

    Paramètres
    ----------
    s : list
        [bitboard noir, bitboard blanc, joueur]
    legal_moves : entier 64 bits
        le bitboard pour les coups légaux

    Retourne
    -------
    a : entier 64 bits
        un seul bit correspondant à la position du coup est activé
    """
    # get the move from ai
    global conv, ai_player
    a = ai_player.move(conv.convert(s, 
                input_format='bitboard',
                output_format=ai_player.get_state_input_format()), 
                 conv.convert(legal_moves,
                    input_format='bitboard_single',
                        output_format=\
            ai_player.get_legal_moves_input_format()))
    # convert the move to bitboard
    a = conv.convert(a,
                input_format=\
                ai_player.get_move_output_format(),
                output_format='bitboard_single')
    return a

@app.route('/', methods=['GET', 'POST'])
def start_page():
    """
    La même fonction est utilisée dans la classe Game de game_env
    cette fonction gère toutes les manipulations liées à la conversion des coups

    Paramètres
    ----------
    s : list
        [bitboard noir, bitboard blanc, joueur]
    legal_moves : entier 64 bits
        le bitboard pour les coups légaux

    Retourne
    -------
    a : entier 64 bits
        un seul bit correspondant à la position du coup est activé
    """
    return render_template('game_ui.html',
            players=dict((k, v[0]) for k,v in ai_players.items()))

@app.route('/ai_choice', methods=['POST'])
def ai_choice():
    """
    cette fonction est appelée après la sélection de l'IA contre laquelle jouer
    l'IA déjà initialisée est sélectionnée ici et l'html correspondant
    à la sélection des pions est envoyé d'ici

    Retourne
    -------
    json : json
        le nom de l'IA à afficher, le nouveau html pour #right_panel
    """
    global ai_player
    c = request.form['ai_player']
    d = request.form['difficulty']
    # convert d to percentage with respect to 10
    d = (int(d)-1)/(10-1.0)
    ai_player = ai_players[c][1]
    # initialize with difficulty
    if(c == 'minimax'):
        ai_player = ai_player(depth=int(d*(9-1) + 1))
    elif(c == 'mcts'):
        ai_player = ai_player(n_sim=int(d*(50000-1) + 100))
    else: # random
        ai_player = ai_player()
    # read the html
    with open('templates/coin_choice_btn.html', 'r') as f:
        coin_choice_html = f.read()
    # append html for reset button
    with open('templates/reset.html', 'r') as f:
        coin_choice_html += f.read()
    # return in json format
    return jsonify(ai_player_name=ai_players[c][0], new_html=coin_choice_html)

@app.route('/coin_choice', methods=['POST'])
def coin_choice():
    """
    cette fonction est appelée après que le joueur a choisi quel pion
    utiliser, l'environnement est réinitialisé ici et les données
    du plateau sont retournées au format json

    Retourne
    -------
    json : json
        black_board, white_board, legal_moves, player, 
        done (si la partie est terminée ou non), ai_player_coin (0/1), score_display_html
    """
    global board_state, board_legal_moves, ai_player_coin, env
    # get the color in the ajax call and reset board accordingly
    c = request.form['color']
    # set ai_player color accordingly
    if(c == 'white'):
        ai_player_coin = 0
    elif(c == 'black'):
        ai_player_coin = 1
    else: # c == 'random'
        if(random() < 0.5):
            ai_player_coin = 0
        else:
            ai_player_coin = 1
    # reset the environment
    done = 0
    board_state, board_legal_moves, player = env.reset()
    # read the html to render for score display
    with open('templates/score_display.html', 'r') as f:
        score_display_html = f.read()
    # append the reset button to html
    with open('templates/reset.html', 'r') as f:
        score_display_html += f.read()
    # modify this html if necessary
    if(ai_player_coin == 1):
        score_display_html = score_display_html\
                            .replace('AI (Black)', 'AI (White)')\
                            .replace('You (White)', 'You (Black)')
    # return the boards and other data, html
    return jsonify(black_board=get_set_bits_list(board_state[0]),
                   white_board=get_set_bits_list(board_state[1]),
                   legal_moves=get_set_bits_list(board_legal_moves),
                   player=player, done=done, ai_player_coin=ai_player_coin,
                   score_display_html=score_display_html)

@app.route('/step', methods=['POST'])
def game_step():
    """
    cette fonction est appelée à chaque fois qu'un joueur/IA veut jouer un coup
    sur le plateau, la position passée est dans [0,63] où 0 est le coin inférieur droit

    Retourne
    -------
    json : json
        black_board, white_board, legal_moves, player, done
    """
    global board_state, board_legal_moves
    # play the move chosen by human
    pos = int(request.form['position'])
    # ai steps if pos is -1
    if(pos == -1):
        a = ai_player_move(board_state, board_legal_moves)
    else: # human player move
        a = 1<<pos
    board_state, board_legal_moves, player, done = env.step(board_state, a)
    # return new states
    return jsonify(black_board=get_set_bits_list(board_state[0]),
                   white_board=get_set_bits_list(board_state[1]),
                   legal_moves=get_set_bits_list(board_legal_moves),
                   player=player, done=done)



@app.route('/get_moves/<filename>', methods=['GET'])
def get_moves(filename):
    """
    Récupère le fichier de coups et le renvoie au client

    Paramètres
    ----------
    filename : str
        Le nom du fichier de coups

    Retourne
    -------
    file : fichier
        Le fichier de coups
    """
    file_path = os.path.join('moves', filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "Fichier non trouvé", 404

@app.route('/get_scores/<filename>', methods=['GET'])
def get_scores(filename):
    """
    Récupère le fichier de scores et le renvoie au client

    Paramètres
    ----------
    filename : str
        Le nom du fichier de scores

    Retourne
    -------
    file : fichier
        Le fichier de scores
    """
    file_path = os.path.join('scores', filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "Fichier non trouvé", 404

if __name__ == '__main__':
    app.run(debug=True)