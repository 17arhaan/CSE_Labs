def topological(adList, vertices):
    predecessor = {i: set() for i in range(vertices)}

    for i in range(vertices):
        for j in adList[i]:
            predecessor[j].add(i)

    queue = []
    order = []

    for i in range(vertices):
        if not predecessor[i]:
            queue.append(i)

    while queue:
        queue = queue[::-1]
        curr = queue.pop()
        order.append(curr)
        queue = queue[::-1]

        if len(order) == vertices:
            return order

        for x in adList[curr]:
            predecessor[x].discard(curr)
            if not predecessor[x]:
                queue.append(x)
    return 'cycle in graph'


if __name__ == '__main__':
    vertices = 6
    adList = [[], [], [3], [1], [0, 1], [2, 0]]
    stack = topological(adList, vertices)
    print(stack)

