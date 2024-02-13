
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cmath

def TSB(lamba,theta, m, C, alpha):
    j = 2 * np.pi / lamba
    j2 = j ** 2 
    k = j * np.cos(theta * np.pi / 180)
    TSBtop = k * ( 1 + m * (1 + 2 * j2 * C))
    TSBbase = k + m * (k + (2 * k * j2 * C) + ((complex(0,1)) * j2 * (1/np.tan(alpha))))
    TSB = TSBtop/TSBbase
    return TSB

def TUB(lamba,theta, m, C, alpha):
    j = 2 * np.pi / lamba
    j2 = j ** 2 
    k = j * np.cos(theta * np.pi / 180)
    k2 = k ** 2
    l = j * np.sin(theta * np.pi / 180)
    l2 = l ** 2
    cot = (1/np.tan(alpha))
    TUBtop = complex(0,-1) * cot * ((m * l2) - (k2 * (1 + 0.5 * j2 * m * C)))
    TUBbase1 = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TUBbase2 = ((1/(m*C))+ 0.5 * j2) 
    TUBbase = TUBbase1 * TUBbase2
    TUB = TUBtop/TUBbase
    return TUB

def TVB(lamba,theta, m, C, alpha):
    j = 2 * np.pi / lamba
    j2 = j ** 2 
    k = j * np.cos(theta * np.pi / 180)
    k2 = k ** 2
    l = j * np.sin(theta * np.pi / 180)
    l2 = l ** 2
    cot = (1/np.tan(alpha))
    TVBtop = complex(0,1) * k * l * cot * (1 + m + (0.5 * j2 * C * m))
    TVBbase1 = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TVBbase2 = ((1/(m*C))+ 0.5 * j2) 
    TVBbase = TVBbase1 * TVBbase2
    TVB = TVBtop/TVBbase
    return TVB

def TSC(lamba,theta, m, C, alpha):
    j = 2 * np.pi / lamba
    j2 = j ** 2 
    k = j * np.cos(theta * np.pi / 180)
    k2 = k ** 2
    l = j * np.sin(theta * np.pi / 180)
    l2 = l ** 2
    cot = (1/np.tan(alpha))
    TSCbase = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TSC = k/TSCbase
    return TSC

def TUC(lamba,theta, m, C, alpha):
    j = 2 * np.pi / lamba
    j2 = j ** 2 
    k = j * np.cos(theta * np.pi / 180)
    k2 = k ** 2
    l = j * np.sin(theta * np.pi / 180)
    l2 = l ** 2
    cot = (1/np.tan(alpha))
#    TUCtop = l2* complex(0,1) * cot + k * C
#    TUCbase1 = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
#    TUCbase2 = ((2/(m*C))+ 0.5 * j2) 
    TUCtop1 = C* k * (3 * l2 * m * C + 2 + j2 * m * C)
    TUCtop2 = C * complex(0,1) * 2 * l2 * cot * m 
    TUCtop = TUCtop1 + TUCtop2
    TUCbase1 = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TUCbase2 = (2 + j2 * m * C)
    TUCbase = TUCbase1 * TUCbase2
    TUC = TUCtop/TUCbase
    return TUC

def TVC(lamba,theta, m, C, alpha):
    j = 2 * np.pi / lamba
    j2 = j ** 2 
    k = j * np.cos(theta * np.pi / 180)
    k2 = k ** 2
    l = j * np.sin(theta * np.pi / 180)
    l2 = l ** 2
    cot = (1/np.tan(alpha))
    TVCtop = -k * l * m * C * (2 * complex(0,1) * cot + 3 * k * C)
    TVCbase1 = k + m * (k + (2 * k * j2 * C) + (complex(0,1) * j2 * cot))
    TVCbase2 = ((2+ j2 * m * C))
    TVCbase = TVCbase1 * TVCbase2
    TVC = TVCtop/TVCbase
    return TVC