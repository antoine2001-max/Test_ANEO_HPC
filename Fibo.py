def Fibo(n):
    F0 = 0
    F1 = 1
    for i in range(n):
        f = F1
        F1 = F0+F1
        F0 = f
    return F1


print(Fibo(100))
