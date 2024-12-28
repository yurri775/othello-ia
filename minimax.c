// run this command to build the shared library
// gcc -fPIC -shared -o minimax.dll minimax.c
#include <stdio.h>
#include "game_env_fn.c"

int get_least_significant_set_bit(unsigned long long s){
    int t = 0;
    while(s){
        if(s&1){
            return t;
        }
        s = s >> 1; t++;
    }
    return 0;
}

int board_heuristics(unsigned long long legal_moves, unsigned int get_max,
                     unsigned long long s0, unsigned long long s1,
                     unsigned int self_player){
    // return difference in total coins in next state
    unsigned int b = get_total_set_bits(s0);
    unsigned int w = get_total_set_bits(s1);
    if(self_player){
        return w - b;
    }
    return b - w;
    // if(get_max){
        // return get_total_set_bits(legal_moves);
    // }
    // return -get_total_set_bits(legal_moves);
}

int move(unsigned long long s0, unsigned long long s1, 
          unsigned long long legal_moves, unsigned int current_depth,
          unsigned int get_max, int alpha, int beta, unsigned int player,
          unsigned int max_depth, unsigned int self_player){
    /*
    see the MinimaxPlayer in players.py for the documentation
    s0-> black bitboard, s1->white bitboard,
    legal_moves-> the bitboard for legal moves,
    current_depth-> current depth of recursion, get_max-> if maximizing or not
    alpha-> integer alpha in minimax, beta-> integer beta in minimax,
    player-> current player to play, max_depth-> max depth for recursion
    */
    // some variable definitions
    unsigned long long ns0, ns1, nlegal_moves, m_uint64;
    unsigned int np, done, i, j;
    // get the legal moves
    unsigned int l = get_total_set_bits(legal_moves);
    int h_list[l], moves[l];
    for(int i = 0; i < l; i++){h_list[i] = 0;}
    // assign the individual moves to moves[l]
    get_set_bits_array(legal_moves, moves);
    for(i=0; i < l; i++){
        // play the move in current state
        m_uint64 = 1; m_uint64 = m_uint64 << moves[i];
        step(s0, s1, player, m_uint64, &ns0, &ns1, &nlegal_moves,
             &np, &done);
        if((current_depth < max_depth) && !done){
            h_list[i] = move(ns0, ns1, nlegal_moves, current_depth+1,
                             1-get_max, alpha, beta, np, max_depth,
                             self_player);
        }else{
            h_list[i] = board_heuristics(legal_moves, get_max,
                                         ns0, ns1, self_player);
        }
        // adjust alpha and beta
        if(get_max){
            if(h_list[i] > alpha){alpha = h_list[i];}
        }else{
            if(h_list[i] < beta){beta = h_list[i];}
        }
        if(beta <= alpha){break;}
    }
    // return according to the depth
    if(!current_depth){
        int h_max = h_list[0], idx=0;
        for(j = 1; j < i; j++){
            if(h_list[j] > h_max){
                h_max = h_list[j]; idx=j;
            }
        }
        /*we return an int and not unsigned long long because in some 
        instance of the recursion, we will return alpha and beta; we are trying
        to keep the function signature same throughout; 64 bit conversion can
        be done in the python code anyways*/
        // return get_least_significant_set_bit(moves[idx]);
        return moves[idx];
    }
    // otherwise backprop the alpha/beta
    if(get_max){return alpha;}
    return beta;
}