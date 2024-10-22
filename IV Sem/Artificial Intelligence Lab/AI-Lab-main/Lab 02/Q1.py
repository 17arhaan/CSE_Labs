class Graph:
    def __init__(self, v):
        self.vertices = v
        self.adj_list = {vertex : [] for vertex in range(v)}
    def movement(self, v1, v2):
        self.adj_list[v1].append(v2)
    def print(self):
        print(self.adj_list)

def main():
    a = int(input("Enter Number of Vertices : "))
    g = Graph(a)
    print("Directed Graph : ")
    for i in range(a):
        b = int(input(f"Out Degree for {i} : "))
        for j in range(b):
            c = int(input(f"Entry Number {j+1} for {i} : "))
            g.movement(i, c)
    g.print()

if __name__ == '__main__':
    main()
