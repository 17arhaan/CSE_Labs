def tsp_dp(graph, start):
    memo = {}

    def tsp_helper(current, visited):
        if len(visited) == len(graph) and current == start:
            return graph[current][start]

        if (current, tuple(visited)) in memo:
            return memo[(current, tuple(visited))]

        min_cost = float('inf')

        for neighbor in graph[current]:
            if neighbor not in visited:
                new_visited = visited.union({neighbor})
                cost = graph[current][neighbor] + tsp_helper(neighbor, new_visited)
                min_cost = min(min_cost, cost)

        memo[(current, tuple(visited))] = min_cost
        return min_cost

    min_cost = tsp_helper(start, {start})
    return min_cost

graph = {
    10: {20: 2, 30: 3},
    20: {40: 5, 30: 3},
    30: {50: 6},
    40: {10: 4, 50: 7},
    50: {}
}

start_city = 10
total_cost = tsp_dp(graph, start_city)

if total_cost != float('inf'):
    print(f'Minimum TSP Cost: {total_cost}')
else:
    print('No cycle found.')
