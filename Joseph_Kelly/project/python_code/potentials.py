import numpy as np

def tully1g_adi(x):
    A, B, C, D = 0.01, 1.6, 0.005, 1
    
    if x > 10:
        return -A
    elif x<-17:
        return -A

    else:
        expo = np.exp(2*D*x**2)
        cosh = np.cosh(2*B*x)
        num_val = A**2 * expo
        num = C**2 - num_val
        num += (C**2 + num_val) * cosh
        num = np.sqrt(num)
        denum = np.sqrt(expo*(1+cosh))
        return -num/denum

def tully1e_adi(x):
    return -tully1g_adi(x)

def tully1g_grad_adi(x):
    ##this version has numerator and denominator that both blow up but cancel each other
    A, B, C, D = 0.01, 1.6, 0.005, 1

    if x>10:
        return 0

    elif x<-17:
        return 0

    else:
        numerator = -3 * C**2 * D * x - C**2 * D * x * np.cosh(3 * B * x) * 1/np.cosh(B * x) 
        numerator += 2 * A**2 * B * np.exp(2 * D * x**2) * np.tanh(B * x)

        denominator = np.sqrt(2) * np.sqrt(np.exp(2 * D * x**2) * np.cosh(B * x)**2) 
        denominator *= np.sqrt(C**2 - A**2 * np.exp(2 * D * x**2) + (C**2 + A**2 * np.exp(2 * D * x**2)) * np.cosh(2 * B * x))
        return -numerator/denominator

def tully1e_grad_adi(x):
    return -tully1g_grad_adi(x)

def tully1_dia(x,i,j):
    A, B, C, D = 0.01, 1.6, 0.005, 1
    if i==0 and j==0:
        return A*np.tanh(B*x)
    elif i==1 and j==1:
        return -A*np.tanh(B*x)
    elif (i==0 and j==1) or (i==1 and j==0):
        return C*np.exp(-D*x**2)
    else:
        print('Invalid diabatic matrix element')

def tully1_adi(x,i,j):
    if i!=j:
        return 0
    elif i==0 and j==0:
        return tully1g_adi(x)
    elif i==1 and j==1:
        return -tully1g_adi(x)