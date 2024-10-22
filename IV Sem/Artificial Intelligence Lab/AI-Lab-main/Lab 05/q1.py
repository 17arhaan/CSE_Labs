def uniform_cost(adList, vertices, src, goal):
    visited = set()
    frontier = [(0, src)]
    while frontier:
        curr_cost, curr_node = min(frontier)
        frontier.remove((curr_cost, curr_node))
        if curr_node == goal:
            print(f'cost = {curr_cost}')
            return
        visited.add(curr_node)
        for cost, neighbor in adList[curr_node]:
            if neighbor not in visited and all(neighbor != node 
                for _, node in frontier):
                    frontier.append((curr_cost + cost, neighbor))
            elif any(neighbor == node and cost + curr_cost < c for c, node in frontier):
                frontier = [(c, node) 
                if node != neighbor 
                else (cost + curr_cost, node) for c, node in frontier]

if __name__ == '__main__':
    adList = [[[2, 1], [5, 3]], [[1, 6]], [[4, 1]], [[5, 1], [6, 6], [2, 4]], [[4, 2], [3, 5]], [[3, 6], [6, 2]],
              [[7, 4]]]
    vertices = 7
    uniform_cost(adList, vertices, 0, 6)