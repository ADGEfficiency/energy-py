# import cupy as cp
import numpy as np

from functools import reduce
import operator

board_size = 4

def empty_boards(n_boards=2, N=board_size):
    return np.zeros((n_boards, N, N), dtype=np.int)

def empty_board(N=board_size):
    return np.zeros((N, N), dtype=np.int)

def random_boards(n_boards, N=board_size):
    return np.random.randint(0, 3, (n_boards, N, N), dtype=np.int)

def random_board(N=board_size):
    return np.random.randint(0, 3, (N, N), dtype=np.int)

def add_number(board):
    board = board.copy()
    board = np.reshape(board, (-1, board_size**2) )
    empty_pos = board == 0
    n_empty = np.sum(empty_pos,axis=(1), keepdims=True)
    pos = np.floor(np.random.random((board.shape[0],1))*n_empty).astype(np.int)
    
    e_sum = np.cumsum(empty_pos, axis=1)-1
    e_sum[~empty_pos] = -1
    mask = (e_sum == pos)
    board[mask] = 2
    board = np.reshape(board, (-1, board_size, board_size))
    return board

def move(board, direction):
    board = board.copy()
    horizontal = direction % 2 == 1  
    invert = direction > 2
    
    board[horizontal] = np.transpose(board[horizontal], axes=(0, 2, 1))
    board[invert] = np.flip(board[invert], axis=1)

    board = up(board)

    board[invert] = np.flip(board[invert], axis=1)
    board[horizontal] = np.transpose(board[horizontal], axes=(0,2,1))
    return board

def up(board):
    new_board = up_step(board)
    if (new_board == board).all():
        return new_board
    else:
        return up(new_board)

def up_step(board):
    board = board.copy()
    board = np.moveaxis(board, 0, 2)
    for i in range(board.shape[0] - 1):
        is_same = (board[i] == board[i + 1])
        
        board[i, is_same] = 2 * board[i, is_same]
        board[i + 1, is_same] = 0

        is_empty = (board[i] == 0)
        board[i, is_empty] = board[i + 1, is_empty]
        board[i + 1, is_empty] = 0

    board = np.moveaxis(board, 2, 0)

    return board

class Games:
    def __init__(self, n_boards, N=4):
        self.boards = empty_boards(n_boards=n_boards, N=N)
        self.n_boards = n_boards

    def step(self, actions):
        previous_boards = self.boards
        self.boards = move(self.boards, actions)
        self.boards = add_number(self.boards)
        rewards = self.boards.sum(axis=(1,2)) #+ self.boards.max(axis=(1,2))
        #is_game_over = (self.boards == previous_boards).all(axis=(1, 2))
        #is_game_over = np.zeros_like(is_game_over, dtype=np.bool)

        def stuck(direction):
            return (self.boards == move(self.boards, direction*np.ones_like(actions))).all(axis=(1,2))

        is_game_over = reduce(
            np.logical_and,
            map(stuck, range(4)),
            np.ones_like(actions)
        )

        return self.boards, rewards, is_game_over, {}

def test_empty():
    empty = empty_boards()
    assert((empty == up(empty)).all())

def test_single_edge():
    single_edge = empty_boards()
    single_edge[:, 0, 0] = 2

    assert((single_edge == up(single_edge)).all())

def test_shift():
    falling = empty_boards()
    falling[:, 1, 1] = 2

    landed = empty_boards()
    landed[:, 0, 1] = 2

    assert((landed == up(falling)).all())

def test_squash():
    falling = empty_boards()
    falling[:, [0, 1], 0] = 2

    squashed = empty_boards()
    squashed[:, 0, 0] = 4
    
    assert((squashed == up(falling)).all())

def test_sum_constant():
    for direction in range(0, 4):
        for i in range(32):
            board = random_boards(2)
            direction = np.random.randint(0, 4, size=(2,), dtype=np.int64)
            moved_board =  move(board, direction)
            assert(
                ((board[direction% 2 == 0].sum(axis=1)) == moved_board[direction% 2 == 0].sum(axis=1)).all()
                )
            assert(
                ((board[direction% 2 == 1].sum(axis=2)) == moved_board[direction% 2 == 1].sum(axis=2)).all()
            )

def test_add_number():
    empty = empty_boards(2)

    single_num = add_number(empty)

    print(single_num)

    assert((np.sum(single_num, axis=(1,2))==2 ).all())

    multiple_num = empty_boards(2)

    for i in range(17):
        assert(
            (np.sum(multiple_num, axis=(1,2)) == np.minimum(2*i, 2*board_size**2 )).all()
        )
        multiple_num = add_number(multiple_num)
        print(i, multiple_num)

def test():
    test_empty()
    test_single_edge()
    test_shift()
    test_squash()
    test_sum_constant()

    test_add_number()

def main():

    import time

    start = time.time()
    n_boards = 1
    game = Games(n_boards)
    board = game.boards

    board, reward = game.step(np.array([0]))
    print(board[0])
    board, reward = game.step(np.array([0]))
    print(board[0])
    board, reward = game.step(np.array([0]))
    print(board[0])
    board, reward = game.step(np.array([0]))
    print(board[0])

    print(time.time() - start)

#main()
