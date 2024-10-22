class Graph:
    def __init__(self, v):
        self.vertices = v
        self.adj_list = {vertex : [] for vertex in range(v)}
    def movement(self, v1, v2, w):
        string = f"{v2},{w}"
        self.adj_list[v1].append(string)
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
            d = int(input("Weight for Movement : "))
            g.movement(i, c, d)
    g.print()
if __name__ == '__main__':
    main()