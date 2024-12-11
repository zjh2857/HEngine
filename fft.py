import numpy as np
e = 2.718281828459045
pi = 3.1415926
def fft(A):    # A 为多项式的点值表达，求出 A 在 n 次单位根的值
    n = len(A)
    
    if n==1:
        return [A[0]]
    
    A0 = fft(A[0::2])  # A0[k] 是 A0((W_nk)^2) 的值
    A1 = fft(A[1::2])
    
    res = np.zeros(n, dtype = np.complex_)
    
    for k in range(n//2):
        x = e^(k/n * 2 * pi * I)
        
        res[k] = A0[k] + x * A1[k]
        res[k+n//2] = A0[k] - x * A1[k]
    
    return res

r = fft([1, 2, 3, 4, 5, 6, 0, 0])
print(r.round(3))
# [21.   +0.j -9.657+3.j  3.   +4.j  1.657-3.j -3.   +0.j  1.657+3.j
#  3.   -4.j -9.657-3.j]

# f(x) = 1 + 2*x + 3*x^2 + 4*x^3 + 5*x^4 + 6*x^5
# y = [f(e^(k/8 * 2 * pi * I)) for k in range(8)]
# print(np.array(y, dtype=complex).round(3))
# # [21.   +0.j -9.657+3.j  3.   +4.j  1.657-3.j -3.   +0.j  1.657+3.j
# #  3.   -4.j -9.657-3.j]