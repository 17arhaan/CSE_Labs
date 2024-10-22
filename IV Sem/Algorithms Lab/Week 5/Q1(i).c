#include <stdio.h>
#include <stdlib.h>
struct Node {
    char val;
    struct Node* next;
};
void addNode(struct Node* head, char value) {
    struct Node* temp = (struct Node*)malloc(sizeof(struct Node));
    temp->val = value;
    temp->next = NULL;
    struct Node* t = head;
    while (t->next != NULL) {
        t = t->next;
    }
    t->next = temp;
}
void printLinkedList(struct Node* head) {
    struct Node* temp = head;
    printf("%c->", temp->val);
    temp = temp->next;
    while (temp != NULL) {
        printf(" |%c| ", temp->val);
        temp = temp->next;
    }
}
struct Graph {
    char name;
    struct Node* list;
    int visited;
};
void dfs(struct Graph* g, int n, char vertex, char* stack, int* top) {
    g[vertex - 'A'].visited = 1;
    struct Node* temp = g[vertex - 'A'].list->next;
    while (temp != NULL) {
        if (!g[temp->val - 'A'].visited) {
            dfs(g, n, temp->val, stack, top);
        }
        temp = temp->next;
    }
    stack[++(*top)] = vertex;
}
void topologicalSort(struct Graph* g, int n) {
    char* stack = (char*)malloc(n * sizeof(char));
    int top = -1;
    for (int i = 0; i < n; i++) {
        if (!g[i].visited) {
            dfs(g, n, g[i].name, stack, &top);
        }
    }
    printf("Topological Sort: ");
    while (top >= 0) {
        printf("%c ", stack[top--]);
    }
    printf("\n");
}
int main() {
    int n, out;
    printf("Number of Nodes in Graph: ");
    scanf("%d", &n);
    struct Graph* g = (struct Graph*)malloc(n * sizeof(struct Graph));
    for (int i = 0; i < n; i++) {
        printf("Name for Node %d of Graph: ", (i + 1));
        scanf(" %c", &g[i].name);
        g[i].list = (struct Node*)malloc(sizeof(struct Node));
        g[i].list->val = g[i].name;
        g[i].list->next = NULL;
        g[i].visited = 0;
    }
    for (int i = 0; i < n; i++) {
        printf("OutDegree for %c: ", g[i].name);
        scanf(" %d", &out);
        for (int j = 0; j < out; j++) {
            char a;
            printf("Input %d for %c: ", (j + 1), g[i].name);
            scanf(" %c", &a);
            addNode(g[i].list, a);
        }
    }
    printf("Graph----->\n");
    for (int i = 0; i < n; i++) {
        printLinkedList(g[i].list);
        printf("\n");
    }
    topologicalSort(g, n);
    return 0;
}