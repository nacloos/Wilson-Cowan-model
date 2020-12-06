# References:
# https://github.com/NeuromatchAcademy/course-content/tree/master/tutorials/W3D2_DynamicNetworks
# Wilson H and Cowan J (1972) Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical Journal 1286068-5)
import numpy as np
import scipy.optimize as opt
from collections import namedtuple



params = namedtuple('params', ['tau_E', 'a_E', 'theta_E', 'tau_I', 'a_I', 'theta_I', 'wEE', 'wEI', 'wIE', 'wII', 'I_ext_E', 'I_ext_I'])


def F(x, a, theta):
	# sigmoid
	return 1/(1 + np.exp(-a*(x-theta))) - 1/(1 + np.exp(a*theta))

def F_inv(x, a, theta):
	# inverse sigmoid
	return theta - 1/a*np.log(1/(x + 1/(1 + np.exp(a*theta))) - 1)

def F_der(x, a, theta):
	# derivative of the sigmoid
	return a*np.exp(-a*(x-theta))/(1+np.exp(-a*(x-theta)))**2

def f(x, t, *pars):
	p = params(*pars)
	E, I = x
	dxdt = np.zeros(x.shape)
	dxdt[0] = 1/p.tau_E * (-E + F(p.wEE*E - p.wEI*I + p.I_ext_E, p.a_E, p.theta_E)) # dE/dt
	dxdt[1] = 1/p.tau_I * (-I + F(p.wIE*E - p.wII*I + p.I_ext_I, p.a_I, p.theta_I)) # dI/dt

	# print("t:", t, "exc:", p.wEE*E + p.I_ext_E(t))
	return dxdt

def nullclines(E_range, I_range, p):
	# param E_range: a range of values of E to compute the corresponding I values on the E-nullcline
	I = 1/p.wEI*(p.wEE*E_range - F_inv(E_range, p.a_E, p.theta_E) + p.I_ext_E) # dE/dt = 0
	E = 1/p.wIE*(p.wII*I_range + F_inv(I_range, p.a_I, p.theta_I) - p.I_ext_I) # dI/dt = 0

	return I, E

def find_equilibrium(x0, p):
	res = opt.root(f, x0, args=(0,*p))
	x_eq = res.x
	# check that it is well an equilibrium of the system
	if (f(x_eq, 0, *p) == 0).all():
		return x_eq
	else:
		print("Enable to find an equilibrium from the inital point, obtain f =", f(x_eq, 0, *p))
		return x0


def jac_eigenvalues(E, I, p):
	J = np.zeros((2,2))
	J[0,0] = 1/p.tau_E*(-1 + F_der(p.wEE*E - p.wEI*I + p.I_ext_E, p.a_E, p.theta_E)*p.wEE)
	J[0,1] = -1/p.tau_E*F_der(p.wEE*E - p.wEI*I + p.I_ext_E, p.a_E, p.theta_E)*p.wEI
	J[1,0] = 1/p.tau_I*F_der(p.wIE*E - p.wII*I + p.I_ext_I, p.a_I, p.theta_I)*p.wEI
	J[1,1] = 1/p.tau_I*(-1 - F_der(p.wIE*E - p.wII*I + p.I_ext_I, p.a_I, p.theta_I)*p.wII)

	return np.linalg.eig(J)


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	x = np.linspace(-10, 10)
	plt.plot(x, F(x, 1, 0))
	plt.plot(x, F_der(x, 1, 0))
	plt.show()
