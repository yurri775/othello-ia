// run this command to build the shared library
// gcc -fPIC -shared -o -Wall -Werror mcts.dll mcts.c
#include "game_env_fn.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
struct Node{
    // define all the variables to store here
    // refer to mcts.py for the complete documentation
    unsigned long long s0;
    unsigned long long s1;
    unsigned int p;
    // legal moves
    unsigned long long legal_moves;
    int *legal_moves_set;
    // total legal moves
    unsigned int total_legal_moves;
    // array to store indices of children in node list
    unsigned int *children;
    unsigned int w;
    unsigned int n;
    unsigned int N;
    // since we have shuffled the legal_moves_set, we can use total_children
    // as the idx from where we have to pick the next unexplored move
    unsigned int total_children;
    int m; // move position, not 64 bit one
    unsigned int terminal; // whether leaf node
    unsigned int parent; // index of parent node
};

void node_init(struct Node *node, unsigned long long s0, unsigned long long s1,
               unsigned int p, unsigned long long legal_moves,
               int move, unsigned int terminal,
               unsigned int parent){
    node->s0 = s0; node->s1 = s1; node->p = p; node->legal_moves = legal_moves;
    // calculate the total set bits and prepare the moves list array
    node->total_legal_moves = get_total_set_bits(legal_moves);
    // int arr[node->total_legal_moves]; //doesnt work, use malloc for dynamic
    node->legal_moves_set = malloc((node->total_legal_moves) * sizeof(int));
    get_set_bits_array(legal_moves, node->legal_moves_set);
    // set the parameters to 0 related to ucb
    node->w = 0; node->n = 0; node->N = 0;
    // children array
    node->children = malloc((node->total_legal_moves) * sizeof(unsigned int));
    node->total_children = 0;
    // move
    node->m = move;
    // terminal or leaf node
    node->terminal = terminal;
    // parent
    node->parent = parent;
}

double get_ucb1(struct Node *node){
    double w = node->w, n = node->n, N = node->N;
    double ans = (w/n) + (sqrt(2) * sqrt(log(N)/n));
    return ans;
}


unsigned int move(unsigned long long s0, unsigned long long s1, unsigned int p,
                      unsigned long long legal_moves, unsigned int n_sim){
    /*s0-> black bitboard, s1-> white bitboard, p-> player,
    legal_moves-> legal moves for current state and player,
    n-> total simulations*/
    // prepare the node list, we will refer to all nodes by indices
    // storing pointers creates problems due to repeated memory allocation
    // every loop of simulation creates one new node
    struct Node *node_list = malloc((n_sim+1) * sizeof(struct Node));
    // root is 0th index
    node_init(&node_list[0], s0, s1, p, legal_moves, -1, 0, 0);
    // define some variables to prevent redeclarations
    unsigned long long ns0, ns1, nlegal_moves, m_uint64;
    /*defining a separate variable to store 65 bit version of move m
    is required as otherwise 1<<m is still an int only and the value
    input to the step function will be incorrect, breaking the environment*/
    unsigned int np, done, total_set_bits, 
                idx, curr_idx, next_list_idx=1, node_idx;
    int winner, m;
    // start the simulations
    struct Node *node;
    while(n_sim--){
        /*############################
        ####### Selection Phase ######
        ############################*/
        /*select a node in the tree that is neither a leaf node
        nor fully explored*/
        node = &node_list[0]; node_idx = 0;
        while(1){
            if(node->total_legal_moves != node->total_children || 
               node->terminal){
                // at least one unexplored move is present, stop the
                // selection here
                break;
            }else{
                // since all nodes of previous node were explored at least
                // once, we go to the next level and select the child 
                // with highest ucb1
                double best_ucb1 = -1, ucb1; idx = node->children[0];
                for(int i = 0; i < node->total_children; i++){
                    curr_idx = node->children[i];
                    ucb1 = get_ucb1(&node_list[curr_idx]);
                    if(ucb1 > best_ucb1){
                        best_ucb1 = ucb1;
                        idx = curr_idx;
                    }
                }
                node = &node_list[idx]; node_idx = idx;
            }
        }

        /*############################
        ####### Expansion Phase ######
        ############################*/
        /*select one of the child nodes for this node which is unexplored*/
        if(!node->terminal){
            /*first get a random move from the moves which have not 
            been added to the mcts tree yet*/
            m = node->legal_moves_set[node->total_children];
            m_uint64 = 1; m_uint64 = m_uint64 << m;
            // play the game and add new node to tree
            step(node->s0, node->s1, node->p, m_uint64, &ns0, &ns1, 
                  &nlegal_moves, &np, &done);
            // create the new node
            node_init(&node_list[next_list_idx], ns0, ns1, np, nlegal_moves, 
                      m, done, node_idx);
            // add the idx in this list to the parent's children list
            // also update the related values
            node->children[node->total_children] = next_list_idx;
            node->total_children++;
            // change the node
            node = &node_list[next_list_idx]; node_idx = next_list_idx;
            next_list_idx++;
        }

        /*############################
        ###### Simulation Phase ######
        ############################*/
        /*play till the end by randomly selecting moves starting from the
        newly created node (in case of terminal node this step is skipped)*/
        if(!node->terminal){
            done = 0; s0 = node->s0; s1 = node->s1; p = node->p;
            legal_moves = node->legal_moves;
            while(!done){
                total_set_bits = get_total_set_bits(legal_moves);
                // reinitialize again and again as it's size is variable
                int move_list[total_set_bits];
                get_set_bits_array(legal_moves, move_list);
                // pick a random move, modulo ensures max is not out of array
                m = move_list[rand()%total_set_bits];
                m_uint64 = 1; m_uint64 = m_uint64 << m;
                step(s0, s1, p, m_uint64, &ns0, &ns1, &nlegal_moves, &np, &done);
                s0 = ns0; s1 = ns1; p = np; legal_moves = nlegal_moves;
            }
            winner = get_winner(s0, s1);
        }

        /*############################
        #### Backpropagation Phase ###
        ############################*/
        /*backproagate the winner value from node (from where we started
        to play) to root to update statistical parameters for each node*/
        while(1){
            node->n++;
            // update the value of N in children
            for(int i = 0; i < node->total_children; i++){
                node_list[node->children[i]].N = node->n;
            }
            if(winner != -1 && winner != node->p){
                node->w++;
            }
            // move one level up, root has move = -1
            if(node->m == -1){break;}
            else{
                // node_idx = node->parent;
                node = &node_list[node->parent];
            }
        }
    }


    /*select the best move after the tree has been trained
    here we select the one with most number of plays*/
    // most_plays already initialized above
    idx = 0; unsigned int most_plays = 0;
    for(int i = 0; i < node_list[0].total_children; i++){
        node = &node_list[node_list[0].children[i]];
        if(node->n > most_plays){
            m = node->m;
            most_plays = node->n;
        }
    }
    // free memory
    for(int i = 0; i < next_list_idx; i++){
        free(node_list[i].legal_moves_set);
        free(node_list[i].children);
    }
    free(node_list);

    // convert to 64 bit in python as data type can be modified
    // when passing from C to python
    return m;
}
