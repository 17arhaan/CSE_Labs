def uniform_cost(adList, vertices, src, goal):
    frontier = [(0, src)]
    visit = set()
    while frontier:
        curr_cost, curr_node = frontier.pop(0)
        if curr_node == goal:
            print(f'cost = {curr_cost}')
            return
        visit.add(curr_node)
        for cost, neighbor in adList[curr_node]:
            if neighbor not in visit and all(neighbor != node for _, node in frontier):
                frontier.append((curr_cost + cost, neighbor))
            elif any(neighbor == node and cost + curr_cost < c for c, node in frontier):
                frontier = [(c, node) if node != neighbor else (cost + curr_cost, node) for c, node in frontier]
        frontier.sort()

if __name__ == '__main__':
    adList = [[[8, 2]], [[40, 0], [17, 4]], [[29, 1], [3, 3]], [[11, 0], [46, 5]], [[31, 2], [40, 5], [53, 6]],
              [[15, 6]], [[17, 5]]]
    vertices = 7
    uniform_cost(adList, vertices, 0, 6)