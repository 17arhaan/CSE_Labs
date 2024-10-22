def dfs(adList, visit, curr, goal):
    if curr == goal:
        return True
    if curr in visit:
        return False
    visit.append(curr)
    for x in adList[curr]:
        if dfs(adList, visit, x, goal):
            return True
    return False


if __name__ == '__main__':
    vertices = 20
    adList = [[], [2, 6], [1, 3], [2, 8], [5], [4, 10], [1, 11], [8], [3, 7], [10, 14], [9, 15], [6, 12], [11, 17],
              [14],
              [9, 13, 19], [10, 20], [17], [16, 18], [17, 19], [18, 14], [15]]
    goal = 5
    visit = []
    for i in range(1, vertices + 1):
        if i not in visit:
            if dfs(adList, visit, i, goal):
                print('maze completed')
                # print("Solution --->")
                # print(visit)
                exit()
    print("incomplete")