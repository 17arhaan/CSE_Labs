def cycle(adList, visit, curr):
    if curr in visit:
        return True
    visit.append(curr)
    for x in adList[curr]:
        if cycle(adList, visit, x):
            return True
    return False

if __name__ == "__main__":
    vertices = 4
    adList = [[1, 2], [2], [0, 3], [3]]
    order = [2, 0, 1, 3]
    visit = []
    for i in order:
        if i not in visit:
            if cycle(adList, visit, i):
                print('cycle exsists')
                exit()
    print('cycle does not exsist')