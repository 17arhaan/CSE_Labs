#include <stdio.h>
#include <stdlib.h>
int assignment(int jobs, int person, int **arr, int cost, int *job)
{
    static int min_cost = 1000000000;
    if (person == jobs)
    {
        for (int i = 0; i < jobs; ++i)
        {
            if (job[i] == 0)
            {
                cost += arr[person - 1][i];
                min_cost = min_cost <= cost ? min_cost : cost;
                return min_cost;
            }
        }
    }
    else
    {
        for (int i = 0; i < jobs; ++i)
        {
            if (job[i] == 0)
            {
                job[i] = 1;
                cost += arr[person - 1][i];
                assignment(jobs, person + 1, arr, cost, job);
                cost -= arr[person - 1][i];
                job[i] = 0;
            }
        }
    }
    return min_cost;
}
void main()
{
    printf("enter no. of people\n");
    int p;
    scanf("%d", &p);
    int **arr = (int **)malloc(sizeof(int *) * p);
    for (int i = 0; i < p; ++i)
    {
        arr[i] = (int *)malloc(sizeof(int) * p);
    }
    int *job = (int *)calloc(sizeof(int), p);
    printf("enter weights\n");
    for (int i = 0; i < p; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            scanf("%d", &arr[i][j]);
        }
    }
    printf("min cost = %d", assignment(p, 1, arr, 0, job));
}