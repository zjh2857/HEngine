from sympy.ntheory.modular import isprime
from sympy.ntheory.residue_ntheory import primitive_root 

modulenum = 4
step = 16384 * 4
primes = []
prime = 2 ** 30 // step * step + 1
N = 2048 * 8
for i in range(1000):
    if(isprime(prime)):
        primes.append(prime)
    prime += step
 
primes = primes[:2 * modulenum]


# prime = 2 ** 60 // step * step + 1

Q = 1
for i in range(modulenum):
    Q *= primes[i]

Qmod = []

for i in range(modulenum):
    Qmod.append((Q//primes[i])%primes[1+modulenum])

print("unsigned long long Qmod_t[",modulenum * modulenum,"] ={",str(Qmod)[1:-1],"};")

# print("Qmod",Qmod)

q_hat_inv = []
for i in range(modulenum):
    q_hat_inv.append(pow(Q//primes[i],-1,primes[i]))

print("unsigned long long q_hat_inv_t[",modulenum,"] ={",str(q_hat_inv)[1:-1],"};")

# print("q_hat_inv",q_hat_inv)

# print("=================")
P = 1
for i in range(modulenum):
    P *= primes[i+modulenum]

Pmod = []
for i in range(modulenum):
    for j in range(modulenum):
        Pmod.append((P//primes[j+modulenum])%primes[i])

print("unsigned long long Pmod_t[",modulenum * modulenum,"] ={",str(Pmod)[1:-1],"};")

p_hat_inv = []

for i in range(modulenum):
    p_hat_inv.append(pow(P//primes[i+modulenum],-1,primes[i+modulenum]))

print("unsigned long long p_hat_inv_t[",modulenum,"] ={",str(p_hat_inv)[1:-1],"};")


Pinv = []

for i in range(modulenum):
    Pinv.append(pow(P,-1,primes[i]))

print("unsigned long long Pinv_t[",modulenum,"] ={",str(Pinv)[1:-1],"};")


Ps = []
for i in range(modulenum):
    Ps.append(P%primes[i])

print(Ps)


# print(primes)
qi_qj_inv = []
for i in range(modulenum):
    # print(qi_qj_inv)
    for j in range(modulenum):
        if i == j:
            qi_qj_inv.append(primes[i]) 
            continue
        qi_qj_inv.append(pow(primes[i],-1,primes[j]))


print("unsigned long long qi_qj_inv[",modulenum * modulenum,"] ={",str(qi_qj_inv)[1:-1],"};")

        