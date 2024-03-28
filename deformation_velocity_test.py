# Deformational velocity test

from calculate_A import *
from calculate_Ud import *
from calculate_eta import *

Hmean = 650
rhoi = 917
g = 9.80665
alpha = 0.02
n = 3

[Amean_linear, Amean_pchip, Amean_akima] = calculate_A_T(Hmean, 'pchip')

Ud_linear = calculate_Ud(Hmean, Amean_linear, rhoi, g, alpha, n)
Ud_pchip = calculate_Ud(Hmean, Amean_pchip, rhoi, g, alpha, n)
Ud_akima = calculate_Ud(Hmean, Amean_akima, rhoi, g, alpha, n)

print('Possible deformational velocities:')
print('Linear Ud: ' + str(round(Ud_linear, 5)) + ' m/yr')
print('PCHIP Ud:  ' + str(round(Ud_pchip, 5)) + ' m/yr')
print('Akima Ud:  ' + str(round(Ud_akima, 5)) + ' m/yr')

[eta_jer_lin, eta_gud_lin] = calculate_eta(Hmean, Amean_linear, rhoi, g, alpha, n, Ud_linear)
[eta_jer_pchip, eta_gud_pchip] = calculate_eta(Hmean, Amean_pchip, rhoi, g, alpha, n, Ud_pchip)
[eta_jer_akima, eta_gud_akima] = calculate_eta(Hmean, Amean_akima, rhoi, g, alpha, n, Ud_akima)

print('')
print('Possible eta values:')
print('Jeremy Linear Eta: ' + str("%.7g" % eta_jer_lin) + '    Gudmundsson Linear Eta: ' + str("%.7g" %(eta_gud_lin*365*24*60*60)))
print('Jeremy PCHIP Eta: ' + str("%.7g" % eta_jer_pchip) + '      Gudmundsson PCHIP Eta: ' + str("%.7g" % (eta_gud_pchip*365*24*60*60)))
print('Jeremy Akima Eta: ' + str("%.7g" % eta_jer_akima) + '      Gudmundsson Akima Eta: ' + str("%.7g" %(eta_gud_akima*365*24*60*60)))