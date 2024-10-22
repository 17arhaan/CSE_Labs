#include <stdio.h>
#include <stdlib.h>
typedef struct adList *al;
struct adList{
	int d;
	al link;
};
typedef struct queue *ql;
struct queue{
	int val;
	ql link;
};
ql create_queue(){
	ql ob = (ql)malloc(sizeof(struct queue));
	ob->link = NULL;
	return ob;
}
void enqueue(ql ob, int ele){
	ql nob = create_queue();
	nob->val = ele;
	while(ob->link != NULL){
		ob = ob->link;
	}
	ob->link = nob;
	nob->link = NULL;
	return;
}
int dequeue(ql ob){
	if(ob->link){
		ql ans = ob->link;
		ob->link = ob->link->link;
		return ans->val;
	}
	return -1;
}
void kahn(al *list, int vertices){
	int visited = 0;
	int *indegree = (int *)calloc(sizeof(int), vertices);
	for(int i = 0; i < vertices; ++i){
		for(al x = list[i]->link; x != NULL; x = x->link){
			++indegree[x->d];
		}
	}
	ql queue = create_queue();
	for(int i = 0; i < vertices; ++i){
		if(indegree[i] == 0){
			enqueue(queue, i);
		}
	}
	while(queue->link != NULL){
		int curr = dequeue(queue);
		printf("%d ", curr);
		++visited;
		if(visited == vertices){
			return;
		}
		for(al x = list[curr]->link; x != NULL; x = x->link){
			if(indegree[x->d] > 0){
				--indegree[x->d];
			}
			if(indegree[x->d] == 0){
				enqueue(queue, x->d);
			}
		}
 
	}
}
void main(){
	printf("enter no. of vertices\n");
    int v;
    scanf("%d", &v);
    al *list = (al *)malloc(sizeof(al) * v);
    for (int i = 0; i < v; ++i)
    {
        list[i] = (al)malloc(sizeof(struct adList));
        list[i]->link = NULL;
    }
    int a;
    do
    {
        printf("enter 1 - edge, 2 - break\n");
        scanf("%d", &a);
        if (a == 1)
        {
            printf("enter source, destination\n");
            int s, d;
            scanf("%d %d", &s, &d);
            al ob = (al)malloc(sizeof(struct adList));
            ob->d = d;
            ob->link = list[s]->link;
            list[s]->link = ob;
        }
    } while (a != 2);
    kahn(list, v);
}
