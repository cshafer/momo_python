import numpy as np
import matplotlib.pyplot as plt

def ept_vs_c(C, time):

    # Investigation of how e^(pt) evolves over time for different C values (which affects p).
    # The difference between the steady state and the transient solutions is by an extra factor
    # that comes into play which is an e^(pt) term. For example, steady state Tsc and transient
    # Tsc is modified by (e^(pt)-1)

    # The time represents how long the perturbation is present at the base and the p value 
    # determines the shape of the decay. If C is larger (large slip ratio ub/ud, VERY SLIPPERY) then the time it 
    # takes to "decay" should take a shorter amount of time whereas if C is smaller (not as slippery, aka stickier?)
    # the the time it takes to decay will take a longer amount of time. The exponential controls how the surface
    # responds in time.  

    alpha = 0.017
    m = 1
    k = 2.308405949243025       # Middle of the k array
    l = 2.308405949243025       # Middle of the l array
    eta = 86949325168547.39
    rhoi = 917
    h_bar = 694.043
    g = 9.80665
    taub = rhoi * g * h_bar * alpha

    j2 = k**2 + l**2
    i = complex(0,1)
    cot = (1/np.tan(alpha))
    xi = (m*C)**(-1) + 2*j2

    p = (-j2*cot/xi) + (k*(C + (xi)**(-1)))*i   # a + bi 
 
    # Non-dimensionalize the time
    time_nd = time*60*60*24*(taub/(2*eta))

    ept_array = []
    for t in time_nd:
        ept = 1 - np.exp(p*t)
        ept_array.append(ept)
    
    return ept_array

time = np.linspace(0, 1, 50)

ept_10 = ept_vs_c(10, time)
ept_5 = ept_vs_c(5, time)
ept_01 = ept_vs_c(.01, time)

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,10))

ax.plot(time, ept_10, 'b-', label = 'C = 10')
ax.plot(time, ept_5, 'r-', label = 'C = 5')
ax.plot(time, ept_01, 'g-', label = 'C = .01')

ax.set_xlabel('time (days)')
ax.set_ylabel('1 - e^(pt)')
ax.set_title('C vs 1-e^(pt)')
ax.legend()

