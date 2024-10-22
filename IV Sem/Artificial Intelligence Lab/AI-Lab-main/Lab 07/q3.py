class Puzzle:
    def __init__(self, i, j, goal, source=None):
        self.matrix = source or [[0] * 3 for _ in range(3)]
        self.blank_i, self.blank_j = i, j
        self.goal = goal
        self.heuristic()

    def matrix_gen(self, src):
        for i in range(3):
            for j in range(3):
                self.matrix[i][j] = src.matrix[i][j]
        self.matrix[src.blank_i][src.blank_j], self.matrix[self.blank_i][self.blank_j] = \
            self.matrix[self.blank_i][self.blank_j], self.matrix[src.blank_i][src.blank_j]

    def heuristic(self):
        self.h = sum(self.matrix[i][j] != self.goal[i][j] for i in range(3) for j in range(3))

def generate(curr, goal):
    possible_pos = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    return [Puzzle(curr.blank_i + x[0], curr.blank_j + x[1], goal, None).matrix_gen(curr) or Puzzle for x in possible_pos if 0 <= curr.blank_i + x[0] <= 2 and 0 <= curr.blank_j + x[1] <= 2]

def a_star(src, goal):
    frontier, visit = [(src.h, src)], set()
    while frontier:
        frontier.sort()
        _, curr = frontier.pop(0)
        if curr.matrix == goal:
            for row in curr.matrix:
                print(row)
            return
        visit.add(tuple(map(tuple, curr.matrix)))
        for x in generate(curr, goal):
            matrix_tuple = tuple(map(tuple, x.matrix))
            if matrix_tuple not in visit:
                frontier.append((x.h, x))

if __name__ == '__main__':
    s = [[1, 2, 3],
         [5, 6, 0],
         [7, 8, 4]]
    g = [[1, 2, 3],
         [5, 8, 6],
         [0, 7, 4]]
    src = Puzzle(1, 2, g, s)
    a_star(src, g)