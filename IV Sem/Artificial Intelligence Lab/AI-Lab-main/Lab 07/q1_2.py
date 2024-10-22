class Graph:
    def __init__(self, v):
        self.vertices = v
        self.adj_list = {v: [] for v in range(self.vertices)}
    
    def movement3(self, v1, v2, w):
        string = f"{v1}, {w}"
        self.adj_list[v1].append((string, v2))  
    
    def movement4(self, v1, v2, w):
        string = f"{v1}, {w}"
        self.adj_list[v1].append((string, v2))
        string = f"{v2}, {w}"
        self.adj_list[v2].append((string, v1))  
    
    def printlist(self):
        for i in self.adj_list:
            print(i, end='-->')
            print(self.adj_list[i])

def heuristic(node, goal):
    return abs(node - goal)

def astar(graph, start, goal):
    visited = set()
    path = []
    queue = [(heuristic(start, goal), start, path)]
    while queue:
        _, current, path = queue.pop(0)
        if current == goal:
            return path + [current]
        if current in visited:
            continue
        visited.add(current)
        for edge in graph.adj_list[current]:
            info, neighbor = edge
            _, cost = info.split(", ")
            new_path = path + [current]
            queue.append((heuristic(neighbor, goal) + int(cost), neighbor, new_path))
        queue.sort(key=lambda x: x[0])
    return None

if __name__ == '__main__':
    a = int(input("Enter Number of Vertices: "))
    g = Graph(a)
    print("Menu\n1.DirectedWeightedGraph\n2.UndirectedWeightedGraph\n")
    b = int(input("Enter Choice: "))
    if b == 1:
        print("Directed Weighted Graph---->\n")
        for i in range(a):
            c = int(input(f"Out Degree for {i}: "))
            for j in range(c):
                d = int(input(f"Entry Number {j+1} for {i}: "))
                e = int(input("Weight for Movement: "))
                g.movement3(i, d, e)
    elif b == 2:
        print("UnDirected Weighted Graph---->\n")
        for i in range(a):
            c = int(input(f"Out Degree for {i}: "))
            for j in range(c):
                d = int(input(f"Entry Number {j+1} for {i}: "))
                e = int(input("Weight for Movement: "))
                g.movement4(i, d, e)
    
    start_node = int(input("Enter the start node: "))
    goal_node = int(input("Enter the goal node: "))
    result = astar(g, start_node, goal_node)
    if result:
        print("Path found:", result)
    else:
        print("No path found")
