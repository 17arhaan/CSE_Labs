#include<stdio.h>
#include<stdlib.h>
struct Heap{
    int *arr;
    int cid;
    int size;
};
struct Heap *heap;
void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
}
int getparent(int idx){
    return (idx-1)/2;
}
int getlchild(int idx){
    return (idx * 2) + 1;
}
int getrchild(int idx){
    return (idx * 2) + 2;
}
void Initialise(){
    int size;
    printf("Input Total Capacity of Heap : ");
    scanf("%d",&size);
    heap = (struct Heap*) malloc(sizeof(struct Heap));
    heap->size = size;
    heap->arr = (int *) malloc(size * sizeof(int));
    heap->cid = 0;
}
void insert(int a){
    if(heap->cid == heap->size){
        printf("Heap Overflow\n");
    }
    else{
        heap->arr[heap->cid] = a;
        int index = heap->cid;
        heap->cid++;
        while(index != 0 && heap->arr[index] < heap->arr[getparent(index)]){
            swap(&heap->arr[index], &heap->arr[getparent(index)]);
            index = getparent(index);
        }
    }
}
void minheapify(int idx){
    int s = idx;
    int ls = getlchild(idx);
    int rs = getrchild(idx);
    if(ls < heap->cid && heap->arr[ls] < heap->arr[s]){
        s = ls;
    }
    if(rs < heap->cid && heap->arr[rs] < heap->arr[s]){
        s = rs;
    }
    if(s != idx){
        swap(&heap->arr[s], &heap->arr[idx]);
        minheapify(s);
    }
}
void maxheapify(int idx){
    int l = idx;
    int ls = getlchild(l);
    int rs = getrchild(l);
    if(ls < heap->cid && heap->arr[ls] > heap->arr[l]){
        l = ls;
    }
    if(rs < heap->cid && heap->arr[rs] > heap->arr[l]){
        l = rs;
    }
    if(l != idx){
        swap(&heap->arr[l], &heap->arr[idx]);
        maxheapify(l);
    }
}
int minhroot(){
    if(heap->cid <= 0){
        return -1;
    }
    int mroot = heap->arr[0];
    heap->arr[0] = heap->arr[--heap->cid];
    minheapify(0);
    return mroot;
}
int maxhroot(){
    if(heap->cid <= 0){
        return -1;
    }
    int mroot = heap->arr[0];
    heap->arr[0] = heap->arr[--heap->cid];
    maxheapify(0);
    return mroot;
}
void PrintMin(){
    while(heap->cid > 0){
        printf("%d ",minhroot());
    }
    printf("\n");
}
void PrintMax(){
    while(heap->cid > 0){
        printf("%d ",maxhroot());
    }
    printf("\n");
}
int main(){
    int a;
    Initialise();
    printf("MinHeap (Enter -1 to Exit)------>\n");
    do{
        printf("Enter Number : ");
        scanf("%d",&a);
        if(a == -1){
            printf("Min Heap ----->\n");
            PrintMin();
        }
        else{
            insert(a);
        }
    }while(a != -1);
    Initialise();
    printf("MaxHeap (Enter -1 to Exit)------>");
    do{
        printf("Enter Number : ");
        scanf("%d",&a);
        if(a == -1){
            printf("Max Heap ----->\n");
            PrintMax();
        }
        else{
            insert(a);
        }
    }while(a != -1);
    return 0;
}