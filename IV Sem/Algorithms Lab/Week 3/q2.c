#include<stdio.h>
#include<string.h>
int opcount = 0;
int BruteForce(char *string, char *pattern){
    int l1, l2;
    l1 = strlen(string);
    l2 = strlen(pattern);
    for(int i=0; i<l1-l2+1; i++){
        int j;
        for(j = 0; j<l2; j++){
            opcount++;
            if(string[i+j] != pattern[j]){
                break;
            }
        }
        if(j == l2){
            return i;
        }
    }
    return -1;
}
int main(){
    char word[100], pattern[20];
    printf("Input Text for Matching : ");
    scanf("%s",&word);
    printf("Input Pattern String to Match : ");
    scanf("%s",&pattern);
    int ans = BruteForce(word, pattern);
    if(ans == -1){
        printf("Pattern Not Found \n");
    }
    else{
        printf("Pattern found at Index : %d",(ans + 1));
        printf("\nOpCount = %d\n",opcount);
    }
    return 0;
}