#include<stdio.h>
#include<stdlib.h>
struct Node
{
    int data;
    struct Node *right, *left;
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
struct Node *lowestCommonAncestor(struct Node *root, struct Node *p, struct Node *q) {
    if (root == NULL || root == p || root == q)
        return root;
    struct Node *left = lowestCommonAncestor(root->left, p, q);
    struct Node *right = lowestCommonAncestor(root->right, p, q);
    if (left != NULL && right != NULL)
        return root;
    return (left != NULL) ? left : right;
}
struct Node *searchKey(struct Node *root, int key) {
    if (root == NULL || root->data == key)
        return root;
    struct Node *leftResult = searchKey(root->left, key);
    if (leftResult != NULL)
        return leftResult;
    return searchKey(root->right, key);
}
void InOrder(struct Node *root){
    if(root != NULL){
        InOrder(root->left);
        printf("%d ",root->data);
        InOrder(root->right);
    }
}
int main(){
    int a, b;
    printf("Root Node (Enter -1 for NULL)\n");
    parent = Create();
    printf("InOrder Traversal---->\n");
    InOrder(parent);
    printf("\nFirst Node Value : ");
    scanf("%d",&a);
    printf("Second Node Value : ");
    scanf("%d",&b);
    struct Node *p = searchKey(parent, a);
    struct Node *q = searchKey(parent, b);
    struct Node *cca = lowestCommonAncestor(parent, p, q);
    printf("\nClosest Common Ancestor of %d and %d is %d\n", p->data, q->data, cca->data);
    return 0;
}