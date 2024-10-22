import random

def f(x):
    return -(x ** 2)

def hill_climbing(max_iterations=1000):
    current_solution = random.uniform(-10, 10)
    current_value = f(current_solution)
    for _ in range(max_iterations):
        neighbor_solution = current_solution + random.uniform(-0.1, 0.1)
        neighbor_solution = max(min(neighbor_solution, 10), -10)
        neighbor_value = f(neighbor_solution)
        if neighbor_value > current_value:
            current_solution = neighbor_solution
            current_value = neighbor_value
    return current_solution, current_value

best_solution, max_value = hill_climbing()
print(max_value, best_solution)
