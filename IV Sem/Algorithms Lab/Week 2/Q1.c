#include<stdio.h>
#include<stdlib.h>
struct node{
	struct node *l;
	struct node *r;
	int data;
};

struct node* create(int data){
	struct node* new=(struct node*)malloc(sizeof(struct node));
	new->l=NULL;
	new->r=NULL;
	new->data=data;
	return new;
}
struct node* insert(struct node* root, int key) {
    if (root == NULL)
        return create(key);

    if (key < root->data)
        root->l = insert(root->l, key);
    else if (key > root->data)
        root->r = insert(root->r, key);

    return root;
}
int search(struct node* root,int x){
	if(root==NULL){
		return -1;
	}
	if(x==root->data){
		return root->data;
	}
	else if(x<=root->data){
		return search(root->l,x);
	}
	else{
		return search(root->r,x);
	}
}
void preorder(struct node* root){
	if(root==NULL){
    return;
	}
	printf("%d",root->data);
	preorder(root->l);
	preorder(root->r);
}
void inorder(struct node* root){
	if(root==NULL){
		return;
	}
	inorder(root->l);
	printf("%d",root->data);
	inorder(root->r);
}
void postorder(struct node* root){
	if(root==NULL){
		return;
	}
	postorder(root->l);
	postorder(root->r);
	printf("%d",root->data); 
}
int main(){
	struct node* root = NULL;
    int n, key;

    printf("Enter the number of elements in the BST: ");
    scanf("%d", &n);

    printf("Enter the elements of the BST:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &key);
        root = insert(root, key);
    }
    int x;
    printf("enter the element to be searched");
    scanf("%d",&x);
    int a=search(root,x);
    if(a!=-1){
    printf("element searched is %d \n",a);
}
    else{
    	insert(root,x);
    	    }
    printf("Inorder traversal: \n ");
    inorder(root);
    printf("\n");

    printf("Postorder traversal: ");
    postorder(root);
    printf("\n");

    printf("Preorder traversal: ");
    preorder(root);
    printf("\n");

    return 0;
}