def binarySearch(arr, x): 
    l = 0
    r = len(arr) - 1
    while l <= r: 
        m = l + (r - l) // 2 
        if arr[m] == x:
            return m
        elif arr[m] < x:
            l = m + 1 
        else:
            r = m - 1
    return -1

if __name__ == "__main__": 
    arr = list(map(int, input("Enter input: ").split()))
    x = int(input("Enter the key: "))
    
    result = binarySearch(arr, x) 

    if result == -1: 
        print("Element not present") 
    else: 
        print("Element found at index", result+1)
