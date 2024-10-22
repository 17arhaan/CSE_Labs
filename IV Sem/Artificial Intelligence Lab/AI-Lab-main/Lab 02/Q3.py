# #undirected unweighted
def add_edge(adj,a,b):
    adj[ord(a)-65].append(b)
    adj[ord(b)-65].append(a)
def vertices(adj):
    vert = []
    for i in range(len(adj)):
        for j in adj[i]:
            if j not in vert:
                vert.append(j)
    return vert
def adj_matrix(adj):
    vertex = vertices(adj)
    matrix = [[0 for i in range(len(vertex))] for i in range(len(vertex))]
    for i in range(len(adj)):
        for item in adj[i]:
            matrix[i][ord(item)-65] = 1
    return matrix
def print_graph(adj):
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            print(f"({chr(i+65)}->{adj[i][j]}",end=")")
        print()
    print()
    mat = adj_matrix(adj)
    for k in mat:
        print(k)

graph = [[] for i in range(5)]
add_edge(graph,'10','B')
add_edge(graph,'A','C')
add_edge(graph,'A','E')
add_edge(graph,'B','C')
add_edge(graph,'C','D')
add_edge(graph,'C','E')
print_graph(graph)
