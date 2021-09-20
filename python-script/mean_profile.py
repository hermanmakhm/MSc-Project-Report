import math
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import sympy as sym
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

ssl = np.genfromtxt('SSL.csv', delimiter=',')
ref = np.genfromtxt('reference.csv', delimiter=',')

#function for curve fitting
def fit(x, a, b, c, q, r, p):
    return (x * (a*(x**2) + b*x + c)/(q*(x**2) + r*x + 1)) + p * (np.log(x+15)-np.log(15))

#symbolic fitted curve for sym.log if needed
def symfit(x, a, b, c, q, r, p):
    return (x * (a*(x**2) + b*x + c)/(q*(x**2) + r*x + 1)) + p * (sym.log(x+15)-sym.log(15))

def loglaw(x, h, A = 5.0, kappa = 0.41):
    return (1/kappa) * np.log(x) + A + h

def symloglaw(x, h, A = 5.0, kappa = 0.41):
    return (1/kappa) * sym.log(x) + A + h

def linlog(x, *h):
    return [loglaw(x[0], h) - x[1], x[0]-x[1]]

plt.figure(figsize=[5,3.75])
#separate arrays and plot digitised points of mean profile
x_ssl = ssl[:,0]
y_ssl = ssl[:,1]
plt.plot(x_ssl, y_ssl, mfc='none', mec='tab:orange',marker='o', markersize=6, linestyle='none', label='SSL')

x_ref = ref[:,0]
y_ref = ref[:,1]
plt.plot(x_ref, y_ref, mfc='none', mec='tab:blue',marker='^', markersize=6, linestyle='none', label='ref')

#establish plotting linspace
xspace_ssl = np.linspace(0.5,109, num=300)
xspace_ref = np.linspace(0.5,198, num=600)

#curve fit and plot
popt_ssl, pcov_ssl = curve_fit(fit, x_ssl, y_ssl, maxfev = 5000)
plt.plot(xspace_ssl, fit(xspace_ssl, *popt_ssl), 'k-.', linewidth=1.5, label='SSL curve fit')

popt_ref, pcov_ref = curve_fit(fit, x_ref, y_ref, maxfev = 5000)
plt.plot(xspace_ref, fit(xspace_ref, *popt_ref), 'k--', linewidth=1.5, label='ref curve fit')

#plot u^+=y^+
plt.plot(np.linspace(0.,30., num=50),np.linspace(0.,30., num=50),'k-', linewidth=1.5, label=r'$\overline{U}^+=y^+$')
plt.xlabel(r'$y^+$')
plt.ylabel(r'$\overline{U}^+$')

#plot graph
plt.legend(loc=4)
plt.grid()

#new graph for estimated log law
plt.figure(figsize=[5,3.75])
plt.plot(x_ssl, y_ssl, mfc='none', mec='tab:orange',marker='o', markersize=6, linestyle='none', label='SSL')
plt.plot(x_ref, y_ref, mfc='none', mec='tab:blue',marker='^', markersize=6, linestyle='none', label='ref')

#plot log law estimates
delh_ssl = 7
plt.plot(xspace_ssl, loglaw(xspace_ssl, delh_ssl), 'k-.', linewidth=1.5, label=r'SSL Log Profile')
plt.plot(xspace_ref, loglaw(xspace_ref, 0), 'k--', linewidth=1.5, label=r'Ref Log Profile')
plt.xlabel(r'$y^+$')
plt.ylabel(r'$\overline{U}^+$')

#plot u^+=y^+
plt.plot(np.linspace(0.,30., num=50),np.linspace(0.,30., num=50),'k-', linewidth=1.5, label=r'$\overline{U}^+=y^+$')
#plot graph
plt.legend(loc=4)
plt.grid()

#symbol for differentiation
x = sym.symbols('x')

#differentiate and square curve fits
diff_ssl = sym.diff(symfit(x, *popt_ssl),x)
diffsr_ssl = diff_ssl**2

diff_ref = sym.diff(symfit(x, *popt_ref),x)
diffsr_ref = diff_ref**2

#plot derivatives squared by lambdifying
diffsr_ssl_lm = sym.lambdify(x, diffsr_ssl, modules='numpy')
diffsr_ref_lm = sym.lambdify(x, diffsr_ref, modules='numpy')
plt.figure(figsize=[5,3.75])
plt.plot(xspace_ssl, diffsr_ssl_lm(xspace_ssl), 'k-.', label='SSL')
plt.plot(xspace_ref, diffsr_ref_lm(xspace_ref), 'k--', label='ref')
plt.grid()
plt.legend()
plt.xlabel(r'$y^+$')
plt.ylabel(r'$\left(\frac{d\overline{U}^+}{dy^+}\right)^2$')
max_range = 60
min_range = 0

#solve where log law estimate profile intersects u^+=y^+
ssl_cross = fsolve(linlog, [11,11], args=(delh_ssl))
ref_cross = fsolve(linlog, [11,11], args=(0))
in_est_ssl = ssl_cross[0] + (1/(.41**2)) * ((1/ssl_cross[0])-(1/max_range))
in_est_ref = ref_cross[0] + (1/(.41**2)) * ((1/ref_cross[0])-(1/max_range))
print(ssl_cross[0],in_est_ssl,in_est_ref)

in_ssl = integrate.quad(diffsr_ssl_lm, min_range, max_range)
in_ref = integrate.quad(diffsr_ref_lm, min_range, max_range)
print(in_ssl,in_ref)
print(in_est_ssl/in_est_ref

r_m = 5.846
C_f0 = 0.0336*(360**(-.273))
k = 0.41
A = 5.0
diss_int = 0.3157
Phi_Ubar_Phi0 = in_est_ssl # or should I use the actual integrated values?
#print(Phi_Ubar_Phi0)
#print("a")
What = np.array([1, 2, 6, 12])
Psav_ssl_coeff = np.array([
    [1.135, 0.002929, -1.205E-6, 1.447E-10, -1.047E-13, 2.609E-17],
    [-1.856, 0.03954, -5.2854E-5, 3.498E-8, -1.127E-11, 1.328E-15],
    [15.25, 0.04888, -4.441E-5, 1.628E-8, -2.845E-12, 1.938E-16],
    [27.90, 0.03824, -2.810E-5, 8.015E-9, -1.082E-12, 5.535E-17]
    ])

def Psav_ssl(lm, coeff):
    a, b, c, d, e, f = coeff
    return a + b * lm + c * (lm**2)+ d * (lm**3) + e * (lm**4) + f * (lm**5)

def Preq_ssl(lm, i):
    return -100*diss_int*math.sqrt(C_f0/2)*(What[i]**2)\
            *((1-Psav_ssl(lm, Psav_ssl_coeff[i])/100)*2*math.pi/lm)**(1/3)

def Pnet_ww_ycross(ycross):
    return 100*(ycross-(1/k)*np.log(ycross)-A)*((2*C_f0)**(-1/2)+k/2)

def Psav_ww(ycross, tau_rat, lm, i):
    #print(math.sqrt(C_f0/2)*(((k*ycross)**2 + 1)/(ycross*(k**2)))*(1-tau_rat/100)**(3/2))
    #print(Psav_ssl(lm, Psav_ssl_coeff[i]))
    return 100*(math.sqrt(C_f0/2)*((((k*ycross)**2 + 1)/(ycross*(k**2)))*.9*(1-tau_rat/100)**(3/2)\
            -in_ssl[0]*(1-Psav_ssl(lm, Psav_ssl_coeff[i])/100)))+Psav_ssl(lm, Psav_ssl_coeff[i])

def P_solve(z, *data):
    index, lmd = data
    Pnw = z[0]
    Psw = z[1]
    ycr = z[2]
    return [Pnw-Psw-r_m*Preq_ssl(lmd, index),
            Pnw-Pnet_ww_ycross(ycr),
            Psw-Psav_ww(ycr, Pnw, lmd, index)]

lamb = np.linspace(250, 3000, num=3000)
ycross_st = np.zeros((4, len(lamb)))
Pnet_ww_st = np.zeros((4, len(lamb)))
Psav_ww_st = np.zeros((4, len(lamb)))
plt.figure()
for i in range(4): 
    for j in range(len(lamb)):
        Pnet_ww_st[i,j], Psav_ww_st[i,j], ycross_st[i,j] = fsolve(P_solve, [0,0,10], args=(i,lamb[j]))
    plt.plot(lamb, Pnet_ww_st[i], label='%i' % What[i])
print(ycross_st)
print(Pnet_ww_st)
plt.legend(title=r"$\hat{W}$")
plt.xlabel(r'$\lambda^{+0}$')
plt.ylabel(r'$P_{net}$')
plt.grid()
plt.show()
