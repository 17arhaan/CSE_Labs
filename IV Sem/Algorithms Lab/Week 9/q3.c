#include <stdio.h>
#include <stdlib.h>
typedef struct hash *hl;
struct hash
{
    int *arr, size;
};
hl create(int size)
{
    hl ob = (hl)malloc(sizeof(struct hash));
    ob->size = size;
    ob->arr = (int *)calloc(sizeof(int), size);
    return ob;
}
void insert(hl ob, int ele)
{
    int index = ele % ob->size;
    if (ob->arr[index] == 0)
    {
        ob->arr[index] = ele;
        return;
    }
    for (int i = index + 1; i != index; i = (i + 1) % ob->size)
    {
        if (ob->arr[i] == 0)
        {
            ob->arr[i] = ele;
            return;
        }
    }
    printf("insertion not possible\n");
    return;
}
void delete(hl ob, int ele)
{
    int index = ele % ob->size;
    if (ob->arr[index] == ele)
    {
        ob->arr[index] = 0;
        return;
    }
    int point = index;
    for (int i = index + 1; i != index; i = (i + 1) % ob->size)
    {
        if (ob->arr[i] == ele)
        {
            ob->arr[i] = 0;
            return;
        }
    }
    printf("deletion not possible\n");
    return;
}
int search(hl ob, int ele)
{
    int index = ele % ob->size;
    if (ob->arr[index] == ele)
    {
        ob->arr[index] = 0;
        return index;
    }
    int point = index;
    for (int i = index + 1; i != index; i = (i + 1) % ob->size)
    {
        if (ob->arr[i] == ele)
        {
            ob->arr[i] = 0;
            return i;
        }
    }
    printf("search insuccessfull\n");
    return -1;
}
void main()
{
    printf("enter size of hash table\n");
    int n;
    scanf("%d", &n);
    hl ob = create(n);
    int a;
    do
    {
        printf("enter 1 for insertion 2 for deletion 3 for search 4 to exit\n");
        scanf("%d", &a);
        if (a == 1)
        {
            printf("enter value\n");
            int v;
            scanf("%d", &v);
            insert(ob, v);
        }
        if (a == 2)
        {
            printf("enter val\n");
            int v;
            scanf("%d", &v);
            delete (ob, v);
        }
        if (a == 3)
        {
            printf("enter val\n");
            int v;
            scanf("%d", &v);
            printf("%d\n", search(ob, v));
        }
    } while (a != 4);
}
