
pi = 3.141592653589793

def Factorial(x):
    res = 1
    for i in range(1,x+1):
        res *= i
    return res

# def talor(q):
#     for i in range(0,31):
#         tmp = 1/Factorial(2*i+1)
#         if i % 2 == 0:
#             coff.append(tmp*q/(2*pi))
#         else:
#             coff.append(-tmp*q/(2*pi))
q = 1156019777784512513
# talor(q)
sgn = []
# print(coff)
for i in range(0,31):
    if(i % 2 == 0):
        sgn.append("+")
    else:
        sgn.append("-")
for i in range(0,31):
    print(f"{sgn[i]}(%.2f*(2*3.141592653589793/{q}/{Factorial(2*i+1)} * x)^{2*i+1})" % (q/(2*pi)),end="")