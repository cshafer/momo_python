
import numpy as np

# Steady state transfer functions in Fourier space

def TSB(k, l, m, C, alpha):
    j2 = k ** 2 + l ** 2
    TSBtop = k * ( 1 + m * (1 + 2 * j2 * C))
    TSBbase = k + m * (k + (2 * k * j2 * C) + ((complex(0,1)) * j2 * (1/np.tan(alpha))))
    TSB = TSBtop/TSBbase
    return TSB

def TUB(k, l, m, C, alpha_s):
    l2 = l**2
    k2 = k ** 2
    j2 = k ** 2 + l ** 2
    cot = (1/np.tan(alpha_s))
    TUBtop = complex(0,-1) * cot * ((m * l2) - (k2 * (1 + 0.5 * j2 * m * C)))
    TUBbase1 = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TUBbase2 = ((1/(m*C))+ 0.5 * j2) 
    TUBbase = TUBbase1 * TUBbase2
    TUB = TUBtop/TUBbase
    return TUB

def TVB(k, l, m, C, alpha_s):
    j2 = l ** 2 + k ** 2 
    cot = (1/np.tan(alpha_s))
    TVBtop = complex(0,1) * k * l * cot * (1 + m + (0.5 * j2 * C * m))
    TVBbase1 = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TVBbase2 = ((1/(m*C))+ 0.5 * j2) 
    TVBbase = TVBbase1 * TVBbase2
    TVB = TVBtop/TVBbase
    return TVB#

def TSC(k, l, m, C, alpha_s):
    j2 = k **2 + l **2 
    cot = (1/np.tan(alpha_s))
    TSCbase = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TSC = k/TSCbase
    return TSC

def TUC(k, l, m, C, alpha_s):
    j2 = l ** 2 + k **2
    l2 = l ** 2
    cot = (1/np.tan(alpha_s))
    TUCtop1 = C* k * (3 * l2 * m * C + 2 + j2 * m * C)
    TUCtop2 = C * complex(0,1) * 2 * l2 * cot * m 
    TUCtop = TUCtop1 + TUCtop2
    TUCbase1 = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TUCbase2 = (2 + j2 * m * C)
    TUCbase = TUCbase1 * TUCbase2
    TUC = TUCtop/TUCbase
    return TUC

def TVC(k, l, m, C, alpha_s):
    j2 = l ** 2 + k ** 2
    cot = (1/np.tan(alpha_s))
    TVCtop = - k * l * m * C * (2 * complex(0,1) * cot + 3 * k * C)
    TVCbase1 = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TVCbase2 = ((2+ j2 * m * C) )
    TVCbase = TVCbase1 * TVCbase2
    TVC = TVCtop/TVCbase
    return TVC