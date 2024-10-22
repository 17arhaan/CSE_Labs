def is_safe(board, row, col):
    for i in range(col):
        if board[row][i] == 1:
            return False

    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    for i, j in zip(range(row, len(board)), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, col, result):
    if col >= len(board):
        result.append([row.index(1) for row in board])
        return

    for i in range(len(board)):
        if is_safe(board, i, col):
            board[i][col] = 1
            solve_n_queens_util(board, col + 1, result)
            board[i][col] = 0

def solve_n_queens():
    n = 8
    board = [[0] * n for _ in range(n)]
    result = []
    solve_n_queens_util(board, 0, result)
    return result

if __name__ == "__main__":
    solutions = solve_n_queens()
    for sol in solutions:
        print("Solution:", sol)
