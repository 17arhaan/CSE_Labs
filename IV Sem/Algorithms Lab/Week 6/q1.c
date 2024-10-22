#include<stdio.h>
#include<stdlib.h>
struct Node
{
    int data;
    struct Node *right;
    struct Node *left;
} *parent = NULL;
struct Node *Create(){
    struct Node *t;
    int a;
    printf("Value for Node : ");
    scanf("%d",&a);
    if (a == -1){
        return NULL;
    }
    else{
        t = (struct Node*) malloc(sizeof(struct Node));
        t->data = a;
        printf("Data for Left of %d Node : ",t->data);
        t->left = Create();
        printf("Data for Right of %d Node : ",t->data);
        t->right = Create();
        return t;
    }
}
int total_nodes(struct Node *root) {
    if (root == NULL){
        return 0;
    }
    else
        return 1 + total_nodes(root->left) + total_nodes(root->right);
}
int main() {
    printf("Root Node : (Enter -1 for NULL)\n");
    parent = Create();
    int total = total_nodes(parent);
    printf("Total Nodes in Tree: %d\n", total);
    return 0;
}