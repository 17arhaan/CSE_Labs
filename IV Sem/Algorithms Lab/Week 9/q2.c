#include <stdio.h>
#include <stdlib.h>
typedef struct Node *hl;
struct Node
{
    int val;
    struct Node *link;
};
struct Node **create(int size)
{
    struct Node **table = (struct Node **)malloc(sizeof(struct Node) * size);
    for (int i = 0; i < size; ++i)
    {
        table[i] = (hl)malloc(sizeof(struct Node));
    }
    return table;
}
void insertion(struct Node **table, int size, int ele)
{
    int index = ele % size;
    struct Node *temp = table[index];
    while (temp->link != NULL)
    {
        temp = temp->link;
    }
    struct Node *ob = (hl)malloc(sizeof(struct Node));
    ob->val = ele;
    temp->link = ob;
    ob->link = NULL;
    return;
}
void deletion(struct Node **table, int size, int ele)
{
    int index = ele % size;
    struct Node *temp = table[index];
    while (temp->link != NULL && temp->link->val != ele)
    {
        temp = temp->link;
    }
    if (temp->link != NULL)
    {
        temp->link = temp->link->link;
    }
    return;
}
int search(struct Node **table, int size, int ele)
{
    int index = ele % size;
    struct Node *temp = table[index];
    temp = temp->link;
    while (temp != NULL)
    {
        if (temp->val == ele)
        {
            return 1;
        }
        temp = temp->link;
    }
    return 0;
}
int main()
{
    struct Node **table = create(10);
    for (int i = 1; i <= 50; ++i)
    {
        insertion(table, 10, i);
    }
    printf("%d ", search(table, 10, 51));
    return 0;
}