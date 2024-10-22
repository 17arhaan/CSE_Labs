import random

def random_state():
    return [random.randint(0, 7) for _ in range(8)]

def conflicts(state):
    return sum(state[i] == state[j] or abs(i - j) == abs(state[i] - state[j]) for i in range(8) for j in range(i + 1, 8))

def successors(state):
    return [[state[i] if j == state[i] else j for j in range(8)] for i in range(8)]

def hill_climbing():
    state = random_state()
    while conflicts(state) > 0:
        best_successor = min(successors(state), key=conflicts)
        if conflicts(best_successor) >= conflicts(state):
            break
        state = best_successor
    return state

def print_solution(state):
    for row in state:
        print(' '.join('Q' if col == row else '.' for col in range(8)))

solution = hill_climbing()
print_solution(solution)
