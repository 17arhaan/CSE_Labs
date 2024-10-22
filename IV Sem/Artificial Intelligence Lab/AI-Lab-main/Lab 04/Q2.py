def bfs(adList, start):
    queue = [start]
    visit = []

    while queue:
        queue = queue[::-1]
        curr = queue.pop()
        queue = queue[::-1]
        visit.append(curr)
        for x in adList[curr]:
            if x in visit:
                return 'cycle exsists'
            queue.append(x)

    return 'no cycle'


if __name__ == '__main__':
    adList = [[1, 2], [2], [0, 3], []]
    print(bfs(adList, 2))