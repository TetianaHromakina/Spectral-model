import numpy as np
import math
from scipy import interpolate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pylab

xnew = np.arange(0.43, 2.28, 0.005)

# Reading data from files
trojan = pylab.loadtxt('agamemnon.dat')
carbon = pylab.loadtxt('carbon.dat')
prx40 = pylab.loadtxt('prx40.dat')
prx80 = pylab.loadtxt('prx80.dat')
olv40 = pylab.loadtxt('olv40.dat')
water = pylab.loadtxt('water.dat')
kerogen = pylab.loadtxt('kerogen.dat')

x_obj = trojan[:,0]
y_obj = trojan[:,1]
f_obj = interpolate.interp1d(x_obj, y_obj)
y_obj_new = f_obj(xnew)

x_carbon = carbon[:,0]
y_carbon = carbon[:,1]
f_carbon = interpolate.interp1d(x_carbon, y_carbon)
y_carbon_new = f_carbon(xnew)

x_prx40 = prx40[:,0]
y_prx40 = prx40[:,1]
f_prx40 = interpolate.interp1d(x_prx40, y_prx40)
y_prx40_new = f_prx40(xnew)

x_prx80 = prx80[:,0]
y_prx80 = prx80[:,1]
f_prx80 = interpolate.interp1d(x_prx80, y_prx80)
y_prx80_new = f_prx80(xnew)

x_olv40 = olv40[:,0]
y_olv40 = olv40[:,1]
f_olv40 = interpolate.interp1d(x_olv40, y_olv40)
y_olv40_new = f_olv40(xnew)

x_water = water[:,0]
y_water = water[:,1]
f_water = interpolate.interp1d(x_water, y_water)
y_water_new = f_water(xnew)

x_ker = kerogen[:,0]
y_ker = kerogen[:,1]
f_ker = interpolate.interp1d(x_ker, y_ker)
y_ker_new = f_ker(xnew)

n_carbon = 1.6
n_prx40 = 1.68
n_prx80 = 1.57
n_olv40 = 1.82
n_water = 1.3
n_ker = 1.5

c_carbon = 0.30
c_prx40 = 0.20
c_prx80 = 0
c_olv40 = 0.1
c_water = 0.0
c_ker = 0.4

q = 0.7

def AlbedoOfSurface(d):
        def opticalDemsity(k, d):
                wavelen = xnew
                tau = 4.0*math.pi*k*d/wavelen
                return tau

        tau_carbon = opticalDemsity(y_carbon_new, d[0])
        tau_prx40 = opticalDemsity(y_prx40_new, d[1])
        tau_prx80 = opticalDemsity(y_prx80_new, d[2])
        tau_olv40 = opticalDemsity(y_olv40_new, d[3])
        tau_water = opticalDemsity(y_water_new, d[4])
        tau_ker = opticalDemsity(y_ker_new, d[5])

        def AlbedoOfParticle(n, tau):
                r0 = (n - 1.0)**2/(n + 1.0)**2
                Re = r0 + 0.05
                Rb = (0.28*n - 0.2)*Re
                Ri = 1.04 - 1/n**2
                Rf = Re - Rb
                Te = 1 - Re
                Ti = 1 - Ri
	
                rb = Rb + 0.5*Te*Ti*Ri*np.exp(-2.0*tau)/(1 - Ri*np.exp(-tau))
                rf = Rf + Te*Ti*np.exp(-tau) + 0.5*Te*Ti*Ri*np.exp(-2.0*tau)/(1 - Ri*np.exp(-tau))
                return rb, rf

        rb_carbon = AlbedoOfParticle(n_carbon, tau_carbon)[0]
        rf_carbon = AlbedoOfParticle(n_carbon, tau_carbon)[1]

        rb_prx40 = AlbedoOfParticle(n_prx40, tau_prx40)[0]
        rf_prx40 = AlbedoOfParticle(n_prx40, tau_prx40)[1]

        rb_prx80 = AlbedoOfParticle(n_prx80, tau_prx80)[0]
        rf_prx80 = AlbedoOfParticle(n_prx80, tau_prx80)[1]

        rb_olv40 = AlbedoOfParticle(n_olv40, tau_olv40)[0]
        rf_olv40 = AlbedoOfParticle(n_olv40, tau_olv40)[1]

        rb_water = AlbedoOfParticle(n_water, tau_water)[0]
        rf_water = AlbedoOfParticle(n_water, tau_water)[1]
        
        rb_ker = AlbedoOfParticle(n_ker, tau_ker)[0]
        rf_ker = AlbedoOfParticle(n_ker, tau_ker)[1]
	
        pb = q*(rb_carbon*c_carbon + rb_prx40*c_prx40 + rb_prx80*c_prx80 + rb_olv40*c_olv40 + rb_water*c_water + rb_ker*c_ker)
        pf = q*(rf_carbon*c_carbon + rf_prx40*c_prx40 + rf_prx80*c_prx80 + rf_olv40*c_olv40 + rf_water*c_water + rf_ker*c_ker) + 1 - q

        A = ((1 + pb**2 - pf**2)/2.0/pb) - np.sqrt(((1 + pb**2 - pf**2)/2.0/pb)**2 - 1)
        
        return sum((A-y_obj_new)**2)

res = minimize(AlbedoOfSurface, [20, 20, 20, 20, 5, 10], method='nelder-mead')
print(res.x)

d = [res.x[0], res.x[1], res.x[2], res.x[3], 15, res.x[5]]

def AlbedoOfSurface1(d):
        def opticalDemsity(k, d):
                wavelen = xnew
                tau = 4.0*math.pi*k*d/wavelen
                return tau

        tau_carbon = opticalDemsity(y_carbon_new, d[0])
        tau_prx40 = opticalDemsity(y_prx40_new, d[1])
        tau_prx80 = opticalDemsity(y_prx80_new, d[2])
        tau_olv40 = opticalDemsity(y_olv40_new, d[3])
        tau_water = opticalDemsity(y_water_new, d[4])
        tau_ker = opticalDemsity(y_ker_new, d[5])

        def AlbedoOfParticle(n, tau):
                r0 = (n - 1.0)**2/(n + 1.0)**2
                Re = r0 + 0.05
                Rb = (0.28*n - 0.2)*Re
                Ri = 1.04 - 1/n**2
                Rf = Re - Rb
                Te = 1 - Re
                Ti = 1 - Ri
	
                rb = Rb + 0.5*Te*Ti*Ri*np.exp(-2.0*tau)/(1 - Ri*np.exp(-tau))
                rf = Rf + Te*Ti*np.exp(-tau) + 0.5*Te*Ti*Ri*np.exp(-2.0*tau)/(1 - Ri*np.exp(-tau))
                return rb, rf

        rb_carbon = AlbedoOfParticle(n_carbon, tau_carbon)[0]
        rf_carbon = AlbedoOfParticle(n_carbon, tau_carbon)[1]

        rb_prx40 = AlbedoOfParticle(n_prx40, tau_prx40)[0]
        rf_prx40 = AlbedoOfParticle(n_prx40, tau_prx40)[1]

        rb_prx80 = AlbedoOfParticle(n_prx80, tau_prx80)[0]
        rf_prx80 = AlbedoOfParticle(n_prx80, tau_prx80)[1]

        rb_olv40 = AlbedoOfParticle(n_olv40, tau_olv40)[0]
        rf_olv40 = AlbedoOfParticle(n_olv40, tau_olv40)[1]

        rb_water = AlbedoOfParticle(n_water, tau_water)[0]
        rf_water = AlbedoOfParticle(n_water, tau_water)[1]

        rb_ker = AlbedoOfParticle(n_ker, tau_ker)[0]
        rf_ker = AlbedoOfParticle(n_ker, tau_ker)[1]
	
        pb = q*(rb_carbon*c_carbon + rb_prx40*c_prx40 + rb_prx80*c_prx80 + rb_olv40*c_olv40 + rb_water*c_water + rb_ker*c_ker)
        pf = q*(rf_carbon*c_carbon + rf_prx40*c_prx40 + rf_prx80*c_prx80 + rf_olv40*c_olv40 + rf_water*c_water + rf_ker*c_ker) + 1 - q

        A = ((1 + pb**2 - pf**2)/2.0/pb) - np.sqrt(((1 + pb**2 - pf**2)/2.0/pb)**2 - 1)
        
        return A

Albedo = AlbedoOfSurface1(d)

def redchisqg(ydata, ymod, sd=None):  
    chisq=np.sum((ydata - ymod)**2/ymod)
    return chisq 

chisq = redchisqg(y_obj_new, Albedo, sd=None)
print("Chi-square value is ", chisq)


for i in Albedo:
    print(i)


pylab.plot(xnew, y_obj_new, 'o', markersize=2)
pylab.plot(xnew, Albedo)
pylab.show()
pylab.close()

