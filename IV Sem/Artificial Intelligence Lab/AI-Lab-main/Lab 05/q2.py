def uniform_cost(adList, vertices, src, goal):
    visited = set()
    queue = [(0, src)]
    while queue:
        queue.sort(key=lambda x: x[0])  # Sort the queue based on cost
        curr_cost, curr_node = queue.pop(0)  # Pop the node with the minimum cost
        if curr_node == goal:
            print(f'cost = {curr_cost}')
            return
        visited.add(curr_node)
        for cost, neighbor in adList[curr_node]:
            if neighbor not in visited:
                queue.append((curr_cost + cost, neighbor))

if __name__ == '__main__':
    adList = [[[5, 1], [9, 2], [6, 4]], [[3, 2], [9, 7]], [[2, 1], [1, 3]], [[6, 0], [5, 8], [7, 6]], [[2, 3], [2, 5]],
              [[7, 9]], [[2, 4], [8, 9]], [], [], []]
    vertices = 10
    uniform_cost(adList, vertices, 0, [7, 8, 9])