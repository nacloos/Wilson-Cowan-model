# References: 
# https://github.com/NeuromatchAcademy/course-content/tree/master/tutorials/W3D2_DynamicNetworks
# Wilson H and Cowan J (1972) Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical Journal 1286068-5)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from collections import namedtuple

params = namedtuple('params', ['tau_E', 'a_E', 'theta_E', 'tau_I', 'a_I', 'theta_I', 'wEE', 'wEI', 'wIE', 'wII', 'I_ext_E', 'I_ext_I'])

def F(x, a, theta):
	# sigmoid
	return 1/(1 + np.exp(-a*(x-theta))) - 1/(1 + np.exp(a*theta))

def F_inv(x, a, theta):
	# inverse sigmoid
	return theta - 1/a*np.log(1/(x + 1/(1 + np.exp(a*theta))) - 1)


def f(x, t, *pars):
	p = params(*pars)
	E, I = x
	dxdt = np.zeros(x.shape)
	dxdt[0] = 1/p.tau_E * (-E + F(p.wEE*E - p.wEI*I + p.I_ext_E, p.a_E, p.theta_E)) # dE/dt
	dxdt[1] = 1/p.tau_I * (-I + F(p.wIE*E - p.wII*I + p.I_ext_I, p.a_I, p.theta_I)) # dI/dt
	return dxdt


def plot_simulation(x0, p, t_end, dt):
	t = np.linspace(0, t_end, int(t_end/dt))
	sol = odeint(f, x0, t, args=p)

	plt.plot(t, sol[:,0], label="E")
	plt.plot(t, sol[:,1], label="I")
	plt.legend()
	plt.xlabel("time")
	plt.ylabel("population activity")


def plot_nullclines(p):
	E_nullcline = np.linspace(-0.01, 0.99, 100)
	I = 1/p.wEI*(p.wEE*E_nullcline - F_inv(E_nullcline, p.a_E, p.theta_E) + p.I_ext_E) # dE/dt = 0

	I_nullcline = np.linspace(-0.1, 0.8, 100)
	E = 1/p.wIE*(p.wII*I_nullcline + F_inv(I_nullcline, p.a_I, p.theta_I) - p.I_ext_I) # dI/dt = 0

	plt.plot(E_nullcline, I, label="$\\frac{dE}{dt} = 0$", color="coral")
	plt.plot(E, I_nullcline, label="$\\frac{dI}{dt} = 0$", color="coral", linestyle="--")
	plt.xlabel("E")
	plt.ylabel("I")
	plt.legend()


def plot_phase_space(p, t_end, dt):
	# E0 = np.linspace(-0.01, 0.99, 15) 
	# I0 = np.linspace(-0.1, 0.8, 15)
	E0 = np.linspace(0.0, 1.0, 15) 
	I0 = np.linspace(0.0, 1.0, 15)

	for i in range(len(E0)):
		for j in range(len(I0)):
			t = np.linspace(0, t_end, int(t_end/dt))
			x0 =  np.array([E0[i], I0[j]])
			sol = odeint(f, x0, t, args=p)

			plt.plot(sol[:,0], sol[:,1], alpha=0.3, color="cornflowerblue")

	plt.xlabel("E")
	plt.ylabel("I")
	plt.legend()



# p = params(tau_E=1., a_E=1.2, theta_E=2.8, tau_I=2., a_I=1.0, theta_I=4.0,
# 		   wEE=9, wEI=4, wIE=13, wII=11, I_ext_E=0, I_ext_I=0)

# Parameters for a limit cycle
p = params(tau_E=1., a_E=1.2, theta_E=2.8, tau_I=2., a_I=3.0, theta_I=3.0,
		   wEE=16, wEI=12, wIE=15, wII=8, I_ext_E=1, I_ext_I=0)


t_end = 50
dt = 1e-1

plt.figure(1)
# plot_phase_space(p, t_end, dt)
plot_nullclines(p)
plt.show()

# plt.figure(2)
# x0 = [0.31, 0.15]
# # x0 = [0.45, 0.25]
# # plot_simulation(x0, p, t_end, dt)
# plot_simulation(x0, p, t_end, dt)
# plt.show()



# param = {'tau_E': 1., 'a_E': 1.2, 'theta_E': 2.8,
# 		 'tau_I': 2., 'a_I': 1.0, 'theta_I': 4.0,
# 		 'wEE': 9, 'wEI': 4, 'wIE': 13, 'wII': 11,
# 		 'I_ext_E': 0, 'I_ext_I': 0}