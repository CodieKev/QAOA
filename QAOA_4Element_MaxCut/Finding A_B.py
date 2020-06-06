
import numpy as np
x = np.arange(0, np.pi, 0.01).tolist()
#print(x)
l = 0
X = [0,0] 
for a_gamma in x:
    for a_beta in x:
        F = (1/32)*(78 - 8*np.cos(a_gamma) + 8*np.cos(3*a_gamma) + 2*np.cos(4*a_gamma) + 10*np.cos(a_gamma - 4*a_beta) + 
   2*np.cos(3*a_gamma - 4*a_beta) + 8*np.cos(2*(a_gamma - 2*a_beta)) - np.cos(4*(a_gamma - a_beta)) + 
   2*np.cos(4*a_beta) - np.cos(4*(a_gamma + a_beta)) - 8*np.cos(2*(a_gamma + 2*a_beta)) - 
   2*np.cos(a_gamma + 4*a_beta) - 10*np.cos(3*a_gamma + 4*a_beta))
        if F>l:
            l = F
            X[0],X[1] = a_gamma,a_beta
            
print(l,X)
