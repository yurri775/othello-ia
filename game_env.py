"""Stores classes for the game environment
and the class to play the game between two players
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import shutil
from tqdm import tqdm
import ctypes
from itertools import chain

def get_set_bits_list(x):
    """returns a list containing the positions of set bits
    Parameters
    ----------
    x : int
        the int for which set bits from binary representation 
        to return
    Returns
    -------
    l : list
        list with positions of set bits, the leftmost bit is position 0
    """
    idx = 0
    """idx represents position from end
    hence bitboard can be prepared by simply shifting 1
    by the idx"""
    l = []
    while(x):
        if(x & 1):
            l.append(idx)
        x = x >> 1
        idx += 1
    return l

def get_random_move_from_list(move_list):
    """Select a random move from a move_list containing
    positions of moves to select from
    Parameters
    ----------
    move_list : list
        list containing positions of moves
    Returns
    -------
    m : int
        position from right hand side where to play the coin
    """
    return move_list[np.random.randint(len(move_list))]

class StateEnv:
    """Base class that implements all the rules of the game
    and also provides the public functions that an agent can
    use to play the game. For determining which player will play,
    white is represented by 1 and black by 0. Note that this class
    has no memory of the game state, and will only perform the task
    of initializing the board, and given an existing state and action,
    perform that action and return the new state, next turn etc.
    The implementation of the game history will be done in a separate
    game class with additional functionality.
    
    Attributes
    ----------
    _size : int
        the size of the board to be played on
    """
    def __init__(self, board_size=8):
        """Initializer for the game environment
        
        Parameters
        ----------
        board_size : int
            kept 8 as the default for othello
        """
        self._size = board_size if board_size%2 == 0 else board_size+1

    def reset(self):
        """Reset the board to starting configuration
        any handicap configurations will go here
        
        Returns
        -------
        starter_board : Ndarray
            the numpy array containing the starting configuration
            of the othello board, positions of black coin are at the
            0th index, white at 1st index and current player at 2nd index
        legal_moves : Ndarray
            masked array denoting the legal positions
        current_player : int
            the integer denoting which player plays
        """
        starter_board = np.zeros((self._size, self._size, 3), 
                                       dtype=np.uint8)
        # put the starting coins
        half_width = (self._size//2) - 1
        # put white coins
        starter_board[[half_width, half_width+1], \
                                    [half_width, half_width+1], 1] = 1
        # put black coins
        starter_board[[half_width, half_width+1], \
                                    [half_width+1, half_width], 0] = 1
        # white to start
        starter_board[:, :, 2] = 1
        # get legal moves
        legal_moves, s = self._legal_moves_helper(starter_board)
        
        return s, legal_moves, 1

    def step(self, s, a):
        """Play a move on the board at the given action location
        and also check for terminal cases, no moves possible etc

        Parameters
        ----------
        s : Ndarray
            current board state
        a : int
            the position where to play the move

        Returns
        -------
        s_next : Ndarray
            updated board
        legal_moves : Ndarray
            legal_moves for the next player
        next_player : int
            whether 1 to play next or 0
        done : int
            1 if the game terminates, else 0
        """
        # variable to track if the game has ended
        # rewards will be determined by the game class
        done = 0
        _, s_next = self._legal_moves_helper(s, 
                           action_row=a//self._size, action_col=a%self._size, play=True)
        # change the player before checking for legal moves
        s_next[:,:,2] = abs(s[0,0,2]-1)
        legal_moves, s_next = self._legal_moves_helper(s_next)
        # check if legal moves are available
        if(legal_moves.sum() == 0):
            # either the current player cannot play, or the game has ended
            if(s_next[:,:,:2].sum() == self._size**2):
                # game has ended
                done = 1
            else:
                # current player cannot play, switch player
                s_next[:,:,2] = s[0,0,2]
                # check for legal moves again
                legal_moves, _ = self._legal_moves_helper(s_next)
                if(legal_moves.sum() == 0):
                    # no moves are possible, game is over
                    done = 1
                else:
                    # original player will play next and opposite player
                    # will pass the turn, nothing to modify
                    pass
        # return
        return s_next, legal_moves, int(s_next[0, 0, 2]), done

    def print(self, s, legal_moves=None):
        """Pretty prints the current board

        Arguments
        ---------
        s : Ndarray
            current board state
        """
        print(('black 0 ' if s[0,0,2]==0 else 'white 1 ') + 'to play')
        s_print = s[:,:,0] * 10 + s[:,:,1]
        print(s_print)
        if(legal_moves is not None):
            print(legal_moves * np.arange(self._size**2).reshape(-1, self._size))
 
    def count_coins(self, s):
        """Count the black and white coins on the board.
        Useful to check winner of the game
        
        Parameters
        ----------
        s : Ndarray
            the board state
        
        Returns
        -------
        (b, w) : tuple
            tuple of ints containing the coin counts
        """
        return (s[:,:,0].sum(), s[:,:,1].sum())

    def _check_legal_index(self, row, col):
        """Check if the row and col indices are possible in the
        current board

        Parameters
        ----------
        row : int
            row index to check
        col : int
            col index to check

        Returns
        -------
        bool : bool
            if the current indices are out of bound
        """
        return 0 <= row and row < self._size and\
               0 <= col and col < self._size

    def _legal_moves_helper(self, s, action_row=None, action_col=None, play=False):
        """A helper function which iterates over all positions on the board
        to check if the position is a legal move or not. Also, if a particular
        action row and action col are specified, this function, instead of checking
        for legal positions, will modify the board according to the game rules
        and return the modified board. If play, the provided action row and col
        are assumed to be legal

        Parameters
        ----------
        s : Ndarray
            current board state
        action_row : int, default None
            the row in which to play a given move
        action_col : int, default None
            the column in which to play a given move

        Returns
        -------
        available_pos : Ndarray
            the mask containing legal moves (all zeros in case a move
            is to be played)
        s : Ndarray
            the modified board state if play is False
        """
        current_player = s[0,0,2]
        opposite_player = 0 if current_player else 1
        # initialize the array of _size to mark available positions
        legal_moves = np.zeros((self._size, self._size), dtype=np.uint8)

        # determine the loop ranges
        row_range = [action_row] if play else range(self._size)
        col_range = [action_col] if play else range(self._size)

        # modify a copy of the board
        s_new = s.copy()

        # loop over all positions to determine if move is legal
        for row in row_range:
            for col in col_range:
                # check if cell is empty
                if(s[row,col,0] + s[row,col,1] == 0):
                    # modify this position
                    if(play):
                        s_new[row, col, current_player] = 1
                    # check the 8 directions for legal moves/modifying position
                    for del_row, del_col in [[-1,-1], [-1,0], [-1,1],
                                             [0,-1], [0,1],
                                             [1,-1], [1,0], [1,1]]:
                        # check if the index is valid
                        n_row, n_col = row+del_row, col+del_col
                        if(self._check_legal_index(n_row, n_col)):
                            # check if the adjacent cell is of the opposite color
                            if(s[n_row, n_col, opposite_player] == 1):
                                # check if moving in this direction continuously will
                                # lead to coin of same color as current player
                                i = 1
                                found = False
                                while(True):
                                    i += 1
                                    n_row, n_col = row+i*del_row, col+i*del_col
                                    if(self._check_legal_index(n_row, n_col)):
                                        # if this cell is blank, break
                                        if(s[n_row, n_col, :2].sum() == 0):
                                            break
                                        # if current player cell encountered again
                                        if(s[n_row, n_col, current_player] == 1):
                                            found = True
                                            break
                                    else:
                                        # we have reached terminal position
                                        break
                                if(found):
                                    # the position is valid, modify on the board
                                    legal_moves[row, col] = 1
                                    # modify the respective positions
                                    if(play):
                                        i = 0
                                        while(True):
                                            i += 1
                                            if(row+i*del_row == n_row and col+i*del_col == n_col):
                                                break
                                            s_new[row+i*del_row, col+i*del_col, opposite_player] = 0
                                            s_new[row+i*del_row, col+i*del_col, current_player] = 1
        # do not change the player in this function, instead do
        # this in the step function to check for potential end conditions
        # if(play):
            # s_new[:,:,2] = opposite_player
        # return the updated boards
        return legal_moves.copy(), s_new.copy()

class StateConverter:
    """Handles conversion between different types of board representations"""
    def __init__(self, board_size=8):
        self._size = board_size
        # a multiplier matrix to convert ndarray to 64 bit int
        self._array_to_bitboard = np.zeros(self._size**2, dtype=np.uint64)
        x = 1
        for i in range(self._array_to_bitboard.shape[0]-1, -1, -1):
            self._array_to_bitboard[i] = x
            x *= 2
        # reshape
        self._array_to_bitboard = self._array_to_bitboard.reshape((-1, self._size))

    def convert(self, s, input_format, output_format):
        """Convert from one board representation to another
        internally, the data is stored as arrays

        Parameters
        ----------
        s : tuple of Ndarray
            input board to convert
        input_format : str
            the input format for board, bitboard or ndarray
        output_format : str
            the output format for board, bitboard or ndarray
        
        Returns
        -------
        s : tuple or Ndarray
            converted board
        """
        # convert input to standard format
        if(input_format == output_format):
            return s
        # internal storage format
        board = np.zeros((self._size, self._size, 3), dtype=np.uint8)

        # convert input format
        if(input_format == 'bitboard'):
            # s is a tuple, convert to arrays
            # iterate to populate values
            for coin in [0, 1]:
                t = s[coin]
                for i in range(self._size-1, -1, -1):
                    for j in range(self._size-1, -1, -1):
                        # same as %2 or get the last set bit
                        board[i, j, coin] = t & 1
                        # shift bits
                        t = t >> 1
                        if(not t):
                            break
                    if(not t):
                        break
            # set the player
            board[:, :, 2] = s[2]
        elif(input_format == 'ndarray3d'):
            # no need to convert
            board = s
        elif(input_format == 'bitboard_single'):
            # the bitboard is a single integer and can be
            # converted to a single array
            for i in range(self._size-1, -1, -1):
                for j in range(self._size-1, -1, -1):
                    board[i, j, 0] = s & 1
                    s = s >> 1
                    if(not s):
                        break
                if(not s):
                    break
        elif(input_format == 'ndarray'):
            # single board is input, example action mask, legal moves
            board[:, :, 0] = s
        elif(input_format == 'position'):
            # integer denoting the position in the grid (row wise expanded)
            # this can be the position where to take the action
            board[s//self._size, s%self._size, 0] = 1
        else:
            print('Input type not understood')

        # convert to output
        if(output_format == 'bitboard'):
            # convert array to bitboard
            board_b = int(np.multiply(board[:, :, 0], self._array_to_bitboard).sum())
            board_w = int(np.multiply(board[:, :, 1], self._array_to_bitboard).sum())
            player = s[0, 0, 2]
            return (board_b, board_w, player)
        elif(output_format == 'ndarray3d'):
            return board.copy()
        elif(output_format == 'ndarray'):
            # 2d array needs to be returned
            return board[:, :, 0].copy()
        elif(output_format == 'bitboard_single'):
            # convert the first array to integer
            return int(np.multiply(board[:, :, 0], self._array_to_bitboard).sum())
        else:
            print('Output type not understood')

class StateEnvBitBoard:
    """Alternate of the StateEnv class that uses bitboards for processing
    to make the gameplay faster. This class will be hardcoded for 8x8 boards
    as 64 bit processing is easier and faster to perform

    Attributes
    _size : int, default 8
        size of the game board (square)
    _max : int
        64 int with all bits 1
    _incorrect_shift_mask : dict
        stores the mask defining legal shifts for every direction
    _bit_shift_fn : dict
        each entry is a function defining bit shifts for every direction
    _directions : list
        list containing strings denoting all directions
    """
    def __init__(self, board_size=8):
        """Initializer for the game environment

        some masks are hardcoded to handle incorrect shifts
        for left shifts, right most column should not be populated
        18374403900871474942 in binary
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0

        for right top shift, left most column and bottom row should not be
        populated, 9187201950435737344 in binary
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 0 0 0 0 0 0 0
        
        Parameters
        ----------
        board_size : int
            kept 8 as the default for othello
        """
        self._size = 8
        # the maximum value
        self._max = 0xFFFFFFFFFFFFFFFF

        # number of possible actions
        self._n_actions = 64

        # define the masks for incorrect bit shifts
        # we can take direct & mask with these values instead
        # of doing & ~mask
        self._incorrect_shift_mask = {
          'left'         : 18374403900871474942,
          'right'        : 9187201950435737471,
          'top'          : 18446744073709551360,
          'bottom'       : 72057594037927935,
          'right_top'    : 9187201950435737344,
          'left_top'     : 18374403900871474688,
          'right_bottom' : 35887507618889599,
          'left_bottom'  : 71775015237779198
        }

        # handy functions to apply bit shifts for different directions
        self._bit_shift_fn = {
            'left'         : lambda x: x << 1,
            'right'        : lambda x: x >> 1,
            'top'          : lambda x: x << 8,
            'bottom'       : lambda x: x >> 8,
            'right_top'    : lambda x: x << 7,
            'left_top'     : lambda x: x << 9,
            'right_bottom' : lambda x: x >> 9,
            'left_bottom'  : lambda x: x >> 7    
        }

        self._directions = self._bit_shift_fn.keys()


    def reset(self):
        """Reset the board to starting configuration
        any handicap configurations will go here
        
        Returns
        -------
        starter_board : Ndarray
            the numpy array containing the starting configuration
            of the othello board, positions of black coin are at the
            0th index, white at 1st index and current player at 2nd index
        legal_moves : Ndarray
            masked array denoting the legal positions
        current_player : int
            the integer denoting which player plays
        """
        starter_b = 34628173824 # check binary arranged as 8x8 to see why
        starter_w = 68853694464 # check binary arranged as 8x8 to see why
        player = 1 # white to start
        starter_board = [starter_b, starter_w, player].copy()
        legal_moves = self._legal_moves_helper(starter_board)
        
        return [starter_board, legal_moves, player]

    def step(self, s, a):
        """Play a move on the board at the given action location
        and also check for terminal cases, no moves possible etc

        Parameters
        ----------
        s : tuple
            current board state defined by bitboards for black, white coins
            and the current player to play
        a : int (64 bit)
            the bit determining the position to play is set to 1

        Returns
        -------
        s_next : list
            updated bitboards for black, white coins, and next player
        legal_moves : bitboard
            legal moves for the next player
        next_player : int
            whether 1 to play next or 0
        done : int
            1 if the game terminates, else 0
        """
        # variable to track if the game has ended
        # rewards will be determined by the game class
        done = 0
        s_next = self._get_next_board(s, a)
        # change the player before checking for legal moves
        s_next[2] = 0 if(s[2]) else 1
        legal_moves = self._legal_moves_helper(s_next)
        # check if legal moves are available
        if(legal_moves == 0):
            # either the current player cannot play, or the game has ended
            if(s_next[0] | s_next[1] == self._max):
                # game has ended
                done = 1
            else:
                # current player cannot play, switch player
                s_next[2] = 1 - s_next[2]
                # check for legal moves again
                legal_moves = self._legal_moves_helper(s_next)
                if(legal_moves == 0):
                    # no moves are possible, game is over
                    done = 1
                else:
                    # original player will play next and opposite player
                    # will pass the turn, nothing to modify
                    pass
        # return
        return s_next, legal_moves, s_next[2], done

    def count_coins(self, s):
        """Count black and white coins on board
         Parameters
         ----------
         s : list
            list containing black bitboard, white bitboard and current player

        Returns
        -------
        w : int
            count of white coins
        b : int
            count of black coins
        """
        return get_total_set_bits(s[0]), get_total_set_bits(s[1])

    def get_player(self, s):
        """return the current player to play

        Parameters
        ----------
        s : tuple
            contains the bitbaords and player

        Returns
        -------
        p : int
        """
        return s[2]

    def get_winner(self, s):
        """Get the winner based on number of coins

        Parameters
        ----------
        s : tuple
            contains the bitboards representing the game state

        Returns
        -------
        w : int
            the winner, 1 if while, 0 if black and -1 if tie
        """
        b, w = self.count_coins(s)
        if(b == w):
            return -1
        if(b > w):
            return 0
        return 1

    def get_num_actions(self):
        """Gets total count of actions in environment"""
        return self._n_actions

    def get_legal_moves(self, s):
        """Get legal moves for the current board state"""
        return self._legal_moves_helper(s)

    def _get_neighbors(self, s, e):
        """Return neighbors of all set bits of input

        Parameters
        ----------
        b : int (64 bit)
            the input bitboard

        e : int (64 bit)
            the bitboard for empty cells

        Returns
        -------
        n : int (64 bit)
            the bitboard with neighbors as set bits
        """
        n = 0
        # take or with all directions
        # for each direction, shift bits and run the mask for incorrect shifts
        for k in self._directions:
            n = n | (self._bit_shift_fn[k](s) & self._incorrect_shift_mask[k])
        # take intersection with empty cells
        n = n & e
        # return
        return n

    def _legal_moves_helper(self, s):
        """Get the bitboard for legal moves of given player

        Parameters
        ----------
        s : list
            contains the bitboard for black coins, white coins and
            the int representing player to play

        Returns
        m : int (64 bit)
            bitboard representing the legal moves for player p
        """
        p = s[2]
        not_p = 0 if(p) else 1
        board_p = s[p]
        board_notp = s[not_p]
        m = 0
        # define the empty set
        e = self._max - (board_p | board_notp)
        # for every direction, run the while loop to get legal moves
        # the while loop traverses paths of same coloured coins
        # get the set of positions where there is a coin of opposite player
        # to the direction of player to play
        """
        for k in self._directions:
            c = board_notp & self._bit_shift_fn[k](board_p) & self._incorrect_shift_mask[k]
            # keep travelling in the same direction until empty space is obtained
            # this will constitute a legal move
            # example : curr op op empty -> empty is legal move
            while(c):
                # if immediately to direction is empty, this is a legal move
                m = m | (e & self._bit_shift_fn[k](c) & self._incorrect_shift_mask[k])
                # we can continue the loop till we keep encountering opposite player
                c = board_notp & self._bit_shift_fn[k](c) & self._incorrect_shift_mask[k]
        # return the completed legal moves list
        """
        # the above code is generic and easy to read
        # following is an implementation to prevent redicrections and loops
        # leads to slightly faster implementation
        # left
        c = board_notp & (board_p << 1) & 18374403900871474942
        while(c):
            # if immediately to direction is empty, this is a legal move
            m = m | (e & (c << 1) & 18374403900871474942)
            # we can continue the loop till we keep encountering opposite player
            c = board_notp & (c << 1) & 18374403900871474942
        # right
        c = board_notp & (board_p >> 1) & 9187201950435737471
        while(c):
            # if immediately to direction is empty, this is a legal move
            m = m | (e & (c >> 1) & 9187201950435737471)
            # we can continue the loop till we keep encountering opposite player
            c = board_notp & (c >> 1) & 9187201950435737471
        # top
        c = board_notp & (board_p << 8) & 18446744073709551360
        while(c):
            # if immediately to direction is empty, this is a legal move
            m = m | (e & (c << 8) & 18446744073709551360)
            # we can continue the loop till we keep encountering opposite player
            c = board_notp & (c << 8) & 18446744073709551360
        # bottom
        c = board_notp & (board_p >> 8) & 72057594037927935
        while(c):
            # if immediately to direction is empty, this is a legal move
            m = m | (e & (c >> 8) & 72057594037927935)
            # we can continue the loop till we keep encountering opposite player
            c = board_notp & (c >> 8) & 72057594037927935
        # right_top
        c = board_notp & (board_p << 7) & 9187201950435737344
        while(c):
            # if immediately to direction is empty, this is a legal move
            m = m | (e & (c << 7) & 9187201950435737344)
            # we can continue the loop till we keep encountering opposite player
            c = board_notp & (c << 7) & 9187201950435737344
        # left_top
        c = board_notp & (board_p << 9) & 18374403900871474688
        while(c):
            # if immediately to direction is empty, this is a legal move
            m = m | (e & (c << 9) & 18374403900871474688)
            # we can continue the loop till we keep encountering opposite player
            c = board_notp & (c << 9) & 18374403900871474688
        # right_bottom
        c = board_notp & (board_p >> 9) & 35887507618889599
        while(c):
            # if immediately to direction is empty, this is a legal move
            m = m | (e & (c >> 9) & 35887507618889599)
            # we can continue the loop till we keep encountering opposite player
            c = board_notp & (c >> 9) & 35887507618889599
        # left_bottom
        c = board_notp & (board_p >> 7) & 71775015237779198
        while(c):
            # if immediately to direction is empty, this is a legal move
            m = m | (e & (c >> 7) & 71775015237779198)
            # we can continue the loop till we keep encountering opposite player
            c = board_notp & (c >> 7) & 71775015237779198
        # return final legal moves
        return m

    def _get_next_board(self, s, a):
        """Determine the updated bitboard after performing action a
        using player passed to the function

        Parameters
        ----------
        s : list
            contains the bitboards for white, black coins and 
            the current player
        a : int (64 bit)
            the bit where action needs to be done is set to 1

        Returns
        -------
        s_next : list
            updated bitboards
        """
        p = s[2]
        not_p = 0 if(p) else 1
        board_p = s[p]
        board_notp = s[not_p]
        # keep a global updates master
        update_master = 0
        # run loops to make the changes
        # this logic is dependent on the fact that in any direction
        # only a single row/diagonal will be modified
        """
        for k in self._directions:
            # define a local update master
            m = 0
            # the immediate neighbor should be of opposite player
            c = board_notp & self._bit_shift_fn[k](a) & self._incorrect_shift_mask[k]
            # keep travelling in the same direction till you encounter same coin
            while(c & board_notp):
                # if c is a coin of current player, we have reached the end
                # if we have reached an empty cell, break and reset m
                # otherwise, continue adding positions to m
                m = m | c
                # update c by shifting, do not check which coin type here
                c = self._bit_shift_fn[k](c) & self._incorrect_shift_mask[k]
            # last encounter after breaking from loop should be same player coin
            if(not(c & board_p)):
                # nothing to modify for this direction
                m = 0
            # update the global master
            update_master = update_master | m
        """
        # the above is an easier to read code
        # below implementation is similar to above, but without any
        # redirections or for loop to slightly increase speed
        # left
        m = 0
        c = board_notp & (a << 1) & 18374403900871474942
        while(c & board_notp):
            m = m | c
            c = (c << 1) & 18374403900871474942
        if(not(c & board_p)):
            m = 0
        else:
            update_master = update_master | m
        # right
        m = 0
        c = board_notp & (a >> 1) & 9187201950435737471
        while(c & board_notp):
            m = m | c
            c = (c >> 1) & 9187201950435737471
        if(not(c & board_p)):
            m = 0
        else:
            update_master = update_master | m
        # top
        m = 0
        c = board_notp & (a << 8) & 18446744073709551360
        while(c & board_notp):
            m = m | c
            c = (c << 8) & 18446744073709551360
        if(not(c & board_p)):
            m = 0
        else:
            update_master = update_master | m
        # bottom
        m = 0
        c = board_notp & (a >> 8) & 72057594037927935
        while(c & board_notp):
            m = m | c
            c = (c >> 8) & 72057594037927935
        if(not(c & board_p)):
            m = 0
        else:
            update_master = update_master | m
        # right_top
        m = 0
        c = board_notp & (a << 7) & 9187201950435737344
        while(c & board_notp):
            m = m | c
            c = (c << 7) & 9187201950435737344
        if(not(c & board_p)):
            m = 0
        else:
            update_master = update_master | m
        # left_top
        m = 0
        c = board_notp & (a << 9) & 18374403900871474688
        while(c & board_notp):
            m = m | c
            c = (c << 9) & 18374403900871474688
        if(not(c & board_p)):
            m = 0
        else:
            update_master = update_master | m
        # right_bottom
        m = 0
        c = board_notp & (a >> 9) & 35887507618889599
        while(c & board_notp):
            m = m | c
            c = (c >> 9) & 35887507618889599
        if(not(c & board_p)):
            m = 0
        else:
            update_master = update_master | m
        # left_bottom
        m = 0
        c = board_notp & (a >> 7) & 71775015237779198
        while(c & board_notp):
            m = m | c
            c = (c >> 7) & 71775015237779198
        if(not(c & board_p)):
            m = 0
        else:
            update_master = update_master | m
        
        # all directions searched, now update the bitboards
        board_p = board_p | update_master | a
        board_notp = board_notp - update_master
        # return
        if(p == 0):
            return [board_p, board_notp, p]
        else:
            return [board_notp, board_p, p]

class StateEnvBitBoardC(StateEnvBitBoard):
    """implementation of parts of StateEnvBitBoard with ctypes and C"""
    def __init__(self, board_size=8):
        StateEnvBitBoard.__init__(self, board_size=board_size)
        # get the dll/so or shared library file
        self._cfns = ctypes.CDLL(os.path.abspath('game_env_fn.dll'))
        """We skip the argument checking for a slight bump in speed
        # assign arguments for the step function
        # we use void function in C and pass the pointers to store
        # the returned objects so that the data type is maintained by python
        self._cfns.step.argtypes = \
            (ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_uint,
             ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_ulonglong),
             ctypes.POINTER(ctypes.c_ulonglong), 
             ctypes.POINTER(ctypes.c_ulonglong),
             ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint))
        # to enforce returning a ulonglong, we will pass the result
        # by reference
        self._cfns.legal_moves_helper.argtypes = \
            (ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_uint,
             ctypes.POINTER(ctypes.c_ulonglong))
        """

    def reset(self):
        starter_b = 34628173824 # check binary arranged as 8x8 to see why
        starter_w = 68853694464 # check binary arranged as 8x8 to see why
        player = 1 # white to start
        starter_board = [starter_b, starter_w, player].copy()
        # setup to call legal_moves_helper
        m = ctypes.c_ulonglong()
        self._cfns.legal_moves_helper(ctypes.c_ulonglong(starter_b), 
                                      ctypes.c_ulonglong(starter_w), 
                                      ctypes.c_uint(player), ctypes.byref(m))
        return [starter_board, m.value, player]

    def step(self, s, a):
        ns0 = ctypes.c_ulonglong()
        ns1 = ctypes.c_ulonglong()
        np = ctypes.c_uint()
        legal_moves = ctypes.c_ulonglong()
        done = ctypes.c_uint()

        self._cfns.step(ctypes.c_ulonglong(s[0]), ctypes.c_ulonglong(s[1]),
                        ctypes.c_uint(s[2]), ctypes.c_ulonglong(a),
                        ctypes.byref(ns0), ctypes.byref(ns1),
                        ctypes.byref(legal_moves), ctypes.byref(np),
                        ctypes.byref(done))
        return [ns0.value, ns1.value, np.value], \
                    legal_moves.value, np.value, done.value

class Game:
    """This class handles the complete lifecycle of a game,
    it keeping history of all the board state, keeps track of two players
    and determines winners, rewards etc
    this class internally stores everything in the bitboard format
    
    Attributes
    _p1 : Player
        the first player
    _p2 : Player
        the second player
    _size : int
        the board size
    _env : StateEnv
        the state env to play game
    _p : dict
        the mapping indicating color of players
    _hist : list
        stores all the state transitions
        [current board, current legal moves, current player, action,
         done, reward, next board state]
    _hist_dict: dict
        mapping for index of stored object in _hist
    """
    def __init__(self, player1, player2, board_size=8):
        """Initialize the game with the two specified players
        the players have a move function which fetches where to play
        the stone of their color next

        Parameters
        ----------
        player1 : Player
            the first player
        player2 : Player
            the second player
        board_size : int, default 8
            the size of board for othello
        """
        self._p1 = player1
        self._p2 = player2
        self._size = board_size
        self._env = StateEnvBitBoardC(board_size=board_size)
        self._converter = StateConverter()
        self._rewards = {'tie':0.5, 'win':1, 'loss':-1}
        self._hist_dict = {'s':0, 'legal_moves':1, 'player':2,
                            'action':3, 'done':4, 'winner':5, 'next_s':6,
                            'next_legal_moves':7, 'next_player':8}
        self.moves = []  # Liste pour mémoriser les coups

    def reset(self):
        self._env.reset()
        self.moves = []  # Réinitialiser la liste des coups

    def play(self, add_to_buffer=False):
        done = False
        while not done:
            current_player = self._p1 if self._env.get_player() == 1 else self._p2
            state, legal_moves = self._env.get_state()
            move = current_player.move(state, legal_moves)
            self.moves.append(move)  # Mémoriser le coup
            state, legal_moves, player, done = self._env.step(state, move)
        return self._env.get_winner()

    def save_moves(self, file_path='moves.txt'):
        with open(file_path, 'w') as f:
            for move in self.moves:
                f.write(f"{move}\n")

    def _flip_vertical(self, x):
        """
        Function for performing vertical flip of the board
        
        Parameters:
        ----------
        x: int (64 bit)
           bitboard

        Returns:
        -------
        x: int (64 bit)
           bitboard of vertical flip of the input bitboard

        """
        k1 = 0x00FF00FF00FF00FF
        k2 = 0x0000FFFF0000FFFF
        k3 = 0x00000000FFFFFFFF
        x = ((x >>  8) & k1) | ((x & k1) <<  8);
        x = ((x >> 16) & k2) | ((x & k2) << 16);
        x = ((x >> 32) & k3) | ((x & k3) << 32);
        
        return x
    
    def _flip_horizontal(self, x):
        """
        Function for performing horizontal flip of the board
        
        Parameters:
        ----------
        x: int (64 bit)
           bitboard

        Returns:
        -------
        x: int (64 bit)
           bitboard of horizontal flip of the input bitboard

        """
        k1 = 0x5555555555555555
        k2 = 0x3333333333333333
        k4 = 0x0f0f0f0f0f0f0f0f
        x = ((x >> 1) & k1) + ((x & k1) << 1);
        x = ((x >> 2) & k2) + ((x & k2) << 2);
        x = ((x >> 4) & k4) + ((x & k4) << 4);
        
        return x
    
    def _flip_diag(self, x):
        """
        Function for performing diagonal flip of the board
        diagonal is line going from (1, 1) to (8, 8)
        
        Parameters:
        ----------
        x: int (64 bit)
           bitboard

        Returns:
        -------
        x: int (64 bit)
           bitboard of diagonal flip of the input bitboard

        """
        k1 = 0x5500550055005500
        k2 = 0x3333000033330000
        k4 = 0x0f0f0f0f00000000
        t  = k4 & (x ^ (x << 28))
        x ^=       t ^ (t >> 28) 
        t  = k2 & (x ^ (x << 14))
        x ^=       t ^ (t >> 14) 
        t  = k1 & (x ^ (x <<  7))
        x ^=       t ^ (t >>  7) 
        return x

    def _flip_anti_diag(self, x):
        """
        Function for performing anti-diagonal flip of the board
        anti-diagonal is line going from (1, 8) to (8, 1)
        
        Parameters:
        ----------
        x: int (64 bit)
           bitboard

        Returns:
        -------
        x: int (64 bit)
           bitboard of anti-diagonal flip of the input bitboard

        """
        k1 = 0xaa00aa00aa00aa00
        k2 = 0xcccc0000cccc0000
        k4 = 0xf0f0f0f00f0f0f0f
        t  =       x ^ (x << 36) 
        x ^= k4 & (t ^ (x >> 36))
        t  = k2 & (x ^ (x << 18))
        x ^=       t ^ (t >> 18) 
        t  = k1 & (x ^ (x <<  9))
        x ^=       t ^ (t >>  9) 
        return x
    
    def _rot_clock_90(self, x):
        """
        Function for performing 90 deg clock-wise rotation of the board
        
        Parameters:
        ----------
        x: int (64 bit)
           bitboard

        Returns:
        -------
        x: int (64 bit)
           bitboard of 90 deg clock-wise rotation of the input bitboard

        """
        return self._flip_diag(self._flip_vertical(x))  
    
    def _rot_180(self, x):
        """
        Function for performing 180 deg rotaion of the board
        
        Parameters:
        ----------
        x: int (64 bit)
           bitboard

        Returns:
        -------
        x: int (64 bit)
           bitboard of 180 deg rotaion of the input bitboard

        """
        return self._flip_horizontal(self._flip_vertical(x))
    
    def _rot_anticlock_90(self, x):
        """
        Function for performing 90 deg anti-clockwise rotaion of the board
        
        Parameters:
        ----------
        x: int (64 bit)
           bitboard

        Returns:
        -------
        x: int (64 bit)
           bitboard of 90 deg anti-clockwise rotaion of the input bitboard

        """
        return self._flip_vertical(self._flip_diag(x))

    def reset(self, player1_coin=None):
        """Randomly select who plays first

        Parameters
        ----------
        player1_coin : int
            if None, do a random assignment
        """
        # prepare object to track history
        self._hist = []
        # random assignment
        r = np.random.random()
        if((player1_coin is None and r < 0.5) or player1_coin==0):
            self._p = {0:self._p1, 1:self._p2}
        else:
            self._p = {1:self._p1, 0:self._p2}

    def get_players_coin(self):
        """return the dictionary telling which player is white/black

        Returns
        -------
        p : dict
            keys are 0,1; items are players
        """
        return self._p


    def play(self, add_to_buffer=False, add_augmentations=False):
        """Play the game to the end

        Returns
        -------
        winner : int
            0 if black wins else 1
        """
        # get the starting state

        s, legal_moves, current_player = self._env.reset()
        self.reset()
        done = 0
        while (not done):
            # get the action from current player
            a = self._p[current_player].move(self._converter.convert(s, 
                                input_format='bitboard',
                                output_format=\
                    self._p[current_player].get_state_input_format()), 
                         self._converter.convert(legal_moves,
                                input_format='bitboard_single',
                                output_format=\
                    self._p[current_player].get_legal_moves_input_format()))
            a = self._converter.convert(a,
                            input_format=\
                    self._p[current_player].get_move_output_format(),
                                        output_format='bitboard_single')
            # step the environment
            next_s, next_legal_moves, next_player, done = \
                            self._env.step(s, a)
            # add to the historybject
            self._hist.append([s, legal_moves, current_player, a, 
                              done, -1, next_s, next_legal_moves, next_player])

            # setup for next iteration of loop
            s = next_s.copy()
            current_player = next_player
            legal_moves = next_legal_moves

        # determine the winner
        winner = self._env.get_winner(s)

        # modify the history object
        self._hist[-1][self._hist_dict['winner']] = winner

        # add game data to buffer agen buffers
        if(add_to_buffer):
            self.game_add_buffer(add_augmentations)

        return winner
    
    
    def evaluate_score(self):
        state, _ = self._env.get_state()
        black_score, white_score = self._env.count_coins(state)
        return black_score, white_score

    def save_score(self, file_path='score.txt'):
        black_score, white_score = self.evaluate_score()
        with open(file_path, 'w') as f:
            f.write(f"Black Score: {black_score}\n")
            f.write(f"White Score: {white_score}\n")

    def game_add_buffer(self, add_augmentations=False):
        """Add game history to players' buffers """
        for player in [0, 1]:
            if(callable(getattr(self._p[player], 'add_to_buffer', None))):
                if(self._hist[-1][self._hist_dict['winner']] == -1):
                    reward = self._rewards['tie']
                elif(self._hist[-1][self._hist_dict['winner']] == player):
                    reward = self._rewards['win']
                else:
                    reward = self._rewards['loss']
                # work with augmentations if needed
                if(add_augmentations):
                    hist_ = self.create_history_augmentations(self._hist)
                else:
                    hist_ = [self._hist]
                for hist_obj in hist_:
                    # get the items relevant to player
                    p_buffer = [item for item in hist_obj \
                                if item[self._hist_dict['player']] == player]
                    # prepare np arrays
                    r = np.zeros((len(p_buffer), 1), dtype=np.float32)
                    r[-1] = reward
                    s = np.array([item[self._hist_dict['s']] \
                                    for item in p_buffer], dtype=np.uint64)
                    a = np.array([item[self._hist_dict['action']] \
                            for item in p_buffer], dtype=np.uint64).reshape(-1, 1)
                    next_s = np.zeros(s.shape, dtype=np.uint64)
                    next_s[1:,:] = s[:-1, :]
                    # done = np.array([item[self._hist_dict['done']] \
                                       # for item in p_buffer]).reshape(-1, 1)
                    # make last entry as done = 1
                    done = np.zeros(r.shape, dtype=np.uint8)
                    done[-1] = 1
                    next_legal = np.array([item[self._hist_dict['next_legal_moves']] \
                                for item in p_buffer], dtype=np.uint64).reshape(-1, 1)
                    self._p[player].add_to_buffer(s, a, r, next_s, \
                                                  done, next_legal)
                                   

    def create_board_reps(self, transition_list):
        """
        Returns a list containing transition list of each transformation of the board
        Total of 20 representation are possible; 4 sides of the board X 5 views - normal, horizontal-flip
        vertical - flip, diagonal flip and anti-diagonal flip
        a lot of these augmentations are redundant and we only return the following 8
        normal, normal rot clock 90, normal rot 180, normal rot anti clock 90
        vertical, vertical rot clock 90, vertical rot 180, vertical rot anti clock 90
        where normal is base transition and vertical is vertical flip

        Parameters:
        ----------
        transition list - list containing s, legal_moves, current_player, a, 
                            done, winner, next_s, next_legal_moves, next_player

        Returns:
        -------
        r - list of transition lists of all representations of the board

        """
        s, legal_moves, current_player, a,\
            done, winner, next_s, next_legal_moves, next_player = transition_list
        
        r = []
        
        for fn_1 in [lambda x: x, self._flip_vertical]:
            # lambda function is for representing normal board state
            for fn_2 in [lambda x: x, self._rot_clock_90, self._rot_180, self._rot_anticlock_90]:
                l = [[fn_2(fn_1(s[0])), fn_2(fn_1(s[1])), s[2]], 
                        fn_2(fn_1(legal_moves)),
                        current_player, fn_2(fn_1(a)),
                        done, winner, 
                     [fn_2(fn_1(next_s[0])), fn_2(fn_1(next_s[1])), next_s[2]],
                     fn_2(fn_1(next_legal_moves)), next_player]
                yield l
        # return modified list
        # return r

    def create_history_augmentations(self, hist_):
        """Create an augmented version of the game history, this function
        returns the original history as well
        
        Parameters
        ----------
        hist_ : list of lists
            a list containing the state variables of an entire game

        Yields
        ------
        hist_aug : list of lists
            one augmentated version of the entire game history
        """
        for fn_1 in [lambda x: x, self._flip_vertical]:
            for fn_2 in [lambda x: x, self._rot_clock_90, self._rot_180,\
                                    self._rot_anticlock_90]:
                hist_aug = []
                for it in hist_:
                    s, legal_moves, current_player, a, done, winner,\
                        next_s, next_legal_moves, next_player = it
                    l = [[fn_2(fn_1(s[0])), fn_2(fn_1(s[1])), s[2]], 
                            fn_2(fn_1(legal_moves)),
                            current_player, fn_2(fn_1(a)),
                            done, winner, 
                         [fn_2(fn_1(next_s[0])), fn_2(fn_1(next_s[1])), next_s[2]],
                         fn_2(fn_1(next_legal_moves)), next_player]
                    hist_aug.append(l.copy())
                # modified hist
                yield hist_aug

    def get_game_history(self, augmentations=False):
        """
        Return the history of the current game played and
        return history with transitions if necessary
        
        Parameters
        ----------
        augmentations : bool
            whether to return history with augmentations

        Returns
        -------
        transition_list : list of lists
            a list containing all history transitions with/without augmentations
        """
        if(not augmentations):
            return self._hist
        else:
            hist_len = len(self._hist)
            transition_list = self._hist.copy()
            for i in range(hist_len):
                transition_list += self.create_board_reps(transition_list[i])
            return transition_list
        
    def record_gameplay(self, path='file.mp4'):
        """Plays a game and saves the frames as individual pngs
    
        Parameters
        ----------
        path : str
            the file name where to save the mp4/gif
        """
        frames_dir = 'temp_frames'
        # color transition from black to white
        transition_color_list = ['forestgreen', 'black', 'dimgray', 'dimgrey', 'gray', 'grey',
                                 'darkgray', 'darkgrey', 'silver', 'lightgray', 
                                 'lightgrey', 'gainsboro', 'whitesmoke', 'white']
        frames_per_anim = len(transition_color_list) - 1
        color_array = np.zeros((self._size, self._size), np.uint8)
        alpha_array = np.zeros((self._size, self._size), np.uint8)
        # play a full game
        winner = self.play()
        # use the history object to save to game
        # temp_frames directory is reserved to save intermediate frames
        if(os.path.exists(frames_dir)):
            # clear out the directory
            # shutil.rmtree(frames_dir)
            for _, _, file_list in os.walk(frames_dir):
                pass
            for f in file_list:
                os.remove(os.path.join(frames_dir, f))
        else:
            os.mkdir(frames_dir)
        # plotting
        ####### Begin Template Creation #######
        # n * h + (n+1) * d = 1, where n is no of cells along 1 axis,
        # h is height of one cell and d is the gap between 2 cells
        delta = 0.005
        cell_height = (1 - ((self._size + 1) * delta))/self._size
        cell_height_half = cell_height/2.0
        # plt x axis runs left to right while y runs bottom to top
        # create the full template for the board here, then just change colors
        # in the loop
        fig, axs = plt.subplots(1, 1, figsize=(8, 8), dpi=72)
        axs.axis('off')
        title_line2 = 'white: ' + \
            str(type(self._p[1])).replace('players.','').replace('<','').replace('>','') + \
                      ' | black: ' + \
          str(type(self._p[0])).replace('players.','').replace('<','').replace('>','')
        if(winner == 1):
            axs.set_title('winner:white\n' + title_line2)
        elif(winner == 0):
            axs.set_title('winner:black\n' + title_line2)
        else:
            axs.set_title('winner:tie\n' + title_line2)

        # add scatter points
        # axs.scatter([0, 1, 0, 1], [0, 1, 1, 0])
        ellipse_patch_list = []
        # add horizontal and vertical lines
        for i in range(self._size):
            # linewidth is dependent on axis size and hence needs
            # to be set manually
            axs.axvline((2 * i + 1)*(delta/2) + i * cell_height, 
                        color='white', lw=2)
            axs.axhline((2 * i + 1)*(delta/2) + i * cell_height, 
                        color='white', lw=2)
        for _ in range(self._size):
            ellipse_patch_list.append([0] * self._size)
        # add the large rect determining the board
        rect = Rectangle((delta, delta),
                         width=1 - 2 * delta, 
                         height=1 - 2 * delta,
                         color='forestgreen')
        axs.add_patch(rect)
        # add circle patches
        s = self._converter.convert(self._hist[0][0],
                                    input_format='bitboard',
                                    output_format='ndarray3d')
        # determine the color and alpha values
        color_array[s[:,:,0] == 1] = transition_color_list.index('black')
        color_array[s[:,:,1] == 1] = transition_color_list.index('white')
        alpha_array = (color_array != 0).astype(np.uint8)
        for i in range(self._size):
            for j in range(self._size):
                # i moves along y axis while j along x
                cell_centre = ((j + 1) * delta + (2*j + 1) * cell_height_half,\
                               (self._size - i) * delta + (2*(self._size - i) - 1) * cell_height_half)
                # a circle will be placed where a coin is
                ellipse = Ellipse(cell_centre,
                                  width=((cell_height - delta)),
                                  height=((cell_height - delta)),
                                  angle=0,
                                  color=transition_color_list[color_array[i][j]], 
                                  alpha=alpha_array[i][j])
                ellipse_patch_list[i][j] = ellipse
                # add to the figure
                axs.add_patch(ellipse_patch_list[i][j])
        # save first figure with some persistence
        fig_file_idx = 0
        for idx in range(frames_per_anim):
            if(idx == 0):
                fig.savefig('{:s}/img_{:05d}.png'.format(frames_dir, fig_file_idx), 
                                                        bbox_inches='tight')
            else:
                shutil.copyfile('{:s}/img_{:05d}.png'.format(frames_dir, 0),
                                '{:s}/img_{:05d}.png'.format(frames_dir, fig_file_idx))
            fig_file_idx += 1
        ######## End Template Creation ########
        # iterate over the game frames with animation
        for idx in tqdm(range(len(self._hist))):
            # clear figure
            # plt.cla()
            # get the board from history
            s = self._converter.convert(self._hist[idx][self._hist_dict['s']],
                                    input_format='bitboard',
                                    output_format='ndarray3d')
            next_s = self._converter.convert(
                                 self._hist[idx][self._hist_dict['next_s']],
                                input_format='bitboard',
                                output_format='ndarray3d')
            # prepare a single frame
            for t in range(frames_per_anim):
                # determine the color and alpha values
                # color change from black to white
                color_array[s[:,:,0] * next_s[:,:,1] == 1] = t + 1
                # color change from white to black
                color_array[s[:,:,1] * next_s[:,:,0] == 1] = frames_per_anim - t
                # no coin now and then
                color_array[s[:,:,:2].sum(2) + next_s[:,:,:2].sum(2) == 0] = 0
                # new coin placed
                color_array[(s[:,:,:2].sum(2) == 0) & (next_s[:,:,0] == 1)] = 1
                color_array[(s[:,:,:2].sum(2) == 0) & (next_s[:,:,1] == 1)] = \
                                        len(transition_color_list)-1
                # set alpha array
                alpha_array = (color_array != 0).astype(np.uint8)
                for i in range(self._size):
                    for j in range(self._size):
                        # i moves along y axis while j along x
                        # a circle will be placed where a coin is
                        ellipse_patch_list[i][j].set_color(
                                        transition_color_list[color_array[i][j]])
                        ellipse_patch_list[i][j].set_alpha(alpha_array[i][j])
                        # axs.scatter(5, 5)
                # figure is prepared, save in temp frames directory
                fig.savefig('{:s}/img_{:05d}.png'.format(frames_dir, fig_file_idx), 
                            bbox_inches='tight')
                fig_file_idx += 1
            # add some persistence before placing another new coin
            fig_file_copy_idx = fig_file_idx - 1
            for _ in range(frames_per_anim if idx== len(self._hist)-1\
                            else frames_per_anim//2):
                shutil.copyfile('{:s}/img_{:05d}.png'.format(frames_dir, fig_file_copy_idx),
                '{:s}/img_{:05d}.png'.format(frames_dir, fig_file_idx))
                fig_file_idx += 1
                
        # all frames have been saved, use ffmpeg to convert to movie
        # output frame rate is different to add some persistence
        os.system('ffmpeg -y -framerate {:d} -pattern_type sequence -i "{:s}/img_%05d.png" \
          -c:v libx264 -r {:d} -pix_fmt yuv420p -vf "crop=floor(iw/2)*2:floor(ih/2)*2" {:s}'\
          .format(int(1.5 * frames_per_anim), frames_dir, int(1.5 * frames_per_anim), path))

def get_set_bits_list(x):
    """returns a list containing the positions of set bits

    Parameters
    ----------
    x : int
        the int for which set bits from binary representation 
        to return

    Returns
    -------
    l : list
        list with positions of set bits, the leftmost bit is position 0
    """
    idx = 0
    """idx represents position from end
    hence bitboard can be prepared by simply shifting 1
    by the idx"""
    l = []
    while(x):
        if(x & 1):
            l.append(idx)
        x = x >> 1
        idx += 1
    return l

def get_total_set_bits(x):
    """returns the total count of set bits in the integer
    we use Brian Kernighan's method here

    Parameters
    ----------
    x : int

    Returns
    -------
    t : int
        the total number of set bits in x
    """
    t = 0
    while(x):
        x = x & (x-1)
        t += 1
    return t

def get_random_move_from_list(move_list):
    """Select a random move from a move_list containing
    positions of moves to select from

    Parameters
    ----------
    move_list : list
        list containing positions of moves

    Returns
    -------
    m : int
        position from right hand side where to play the coin
    """
    return move_list[np.random.randint(len(move_list))]

