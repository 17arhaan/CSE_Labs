def ternarySearch(arr, l, r, x):
    if r >= l:
        mid1 = l + (r - l) // 3
        mid2 = r - (r - l) // 3

        if arr[mid1] == x:
            return mid1
        if arr[mid2] == x:
            return mid2

        if x < arr[mid1]:
            return ternarySearch(arr, l, mid1 - 1, x)
        elif x > arr[mid2]:
            return ternarySearch(arr, mid2 + 1, r, x)
        else:
            return ternarySearch(arr, mid1 + 1, mid2 - 1, x)

    return -1

if __name__ == "__main__":
    arr = list(map(int, input("Enter input : ").split()))
    x = int(input("Enter the key: "))
    
    result = ternarySearch(arr, 0, len(arr) - 1, x)
    
    if result == -1:
        print("Element not present")
    else:
        print("Element found at index", result+1)
