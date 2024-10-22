def solve_cryptarithmetic():
    for s in range(1, 10):  
        for e in range(10):
            for n in range(10):
                for d in range(10):
                    for m in range(1, 10):  
                        for o in range(10):
                            for r in range(10):
                                for y in range(10):
                                    if len(set([s, e, n, d, m, o, r, y])) == 8:  
                                        send = s * 1000 + e * 100 + n * 10 + d
                                        more = m * 1000 + o * 100 + r * 10 + e
                                        money = m * 10000 + o * 1000 + n * 100 + e * 10 + y
                                        if send + more == money:
                                            return {
                                                'S': s, 'E': e, 'N': n, 'D': d,
                                                'M': m, 'O': o, 'R': r, 'Y': y
                                            }
    return None

if __name__ == "__main__":
    solution = solve_cryptarithmetic()
    
    if solution:
        for char, digit in solution.items():
            print(f"{char} = {digit}")
    else:
        print("No solution found.")
