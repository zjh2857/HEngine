from math import sin
x = -0.114514
def sinseries(x):
    for i in range(1000):
        x = sin(x*3.1415926/2)
        print(i,x)
        
sinseries(x)