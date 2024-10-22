def enqueue(st1, st2, ele):
    if not st1:
        st2.append(ele)
    else:
        st1.append(ele)
    return


def dequeue(st1, st2):
    if not st1:
        for i in range(len(st2)):
            st1.append(st2.pop())
        ans = st1.pop()
        for i in range(len(st1)):
            st2.append(st1.pop())
        return ans
    else:
        for i in range(len(st1)):
            st2.append(st1.pop())
        ans = st2.pop()
        for i in range(len(st2)):
            st1.append(st2.pop())
        return ans


if __name__ == '__main__':
    a = 0
    st1, st2 = [], []
    while a != 3:
        a = int(input('enter 1 to enqueue 2 to dequeue 3 to exit'))
        if a == 1:
            b = int(input('enter value'))
            enqueue(st1, st2, b)
        if a == 2:
            print(dequeue(st1, st2))