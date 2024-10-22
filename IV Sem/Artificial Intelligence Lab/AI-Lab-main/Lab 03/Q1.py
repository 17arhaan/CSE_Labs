class Graph:
    def topological(self, adList, visit, stack, curr):
        if curr in visit:
            return stack
        visit.append(curr)
        for x in adList[curr]:
            stack = self.topological(adList, visit, stack, x)
        stack.append(curr)
        return stack
    
if __name__ == "__main__":
    vertices = 6
    adList = [[], [], [3], [1], [0, 1], [2, 0]]
    visit = []
    stack = []
    ob = Graph()
    for i in range(vertices):
        if i not in visit:
            stack = ob.topological(adList, visit, stack, i)
    stack = stack[::-1]
    print(stack)
