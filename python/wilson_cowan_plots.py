# References: 
# https://github.com/NeuromatchAcademy/course-content/tree/master/tutorials/W3D2_DynamicNetworks
# Wilson H and Cowan J (1972) Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical Journal 1286068-5)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from wilson_cowan_model import *


t_end = 20
dt = 1e-3

# p = params(tau_E=1., a_E=1.2, theta_E=2.8, tau_I=2., a_I=1.0, theta_I=4.0,
# 		   wEE=9, wEI=4, wIE=13, wII=11, I_ext_E=0.5, I_ext_I=0)

# Parameters for a limit cycle
p = params(tau_E=1., a_E=1.2, theta_E=2.8, tau_I=2., a_I=3.0, theta_I=3.0,
		   wEE=16, wEI=12, wIE=15, wII=8, I_ext_E=1, I_ext_I=0)

# def input_current(t):
# 	R = 7e-2
# 	return R*(t < t_end/2)
# parameters to match network of LIF
# p = params(tau_E=1e-2, a_E=1e4, theta_E=6e-2, tau_I=2., a_I=3.0, theta_I=3.0,
# 		   wEE=1, wEI=0, wIE=0, wII=0, I_ext_E=input_current, I_ext_I=0)


def plot_simulation(x0, p, t_end, dt):
	t = np.linspace(0, t_end, int(t_end/dt))
	sol = odeint(f, x0, t, args=p)

	plt.plot(t, sol[:,0], label="E")
	plt.plot(t, sol[:,1], label="I")
	plt.legend()
	plt.xlabel("time")
	plt.ylabel("population activity")


def plot_nullclines(p):
	E_nullcline = np.linspace(0., 0.99, 100)
	I_nullcline = np.linspace(0., 0.99, 100)
	
	I, E = nullclines(E_nullcline, I_nullcline, p)

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


def plot_eigenvalues(p):
	init_points = [[0,0], [0.4,0.4], [0.93, 0.67]]
	equilibria = []
	eigenvalues = []

	for init_point in init_points:
		E_eq, I_eq = find_equilibrium(init_point, p)
		eig = jac_eigenvalues(E_eq, I_eq, p)[0]
		print("Equilibrium:", E_eq, I_eq, ", eigenvalues:", eig)
		equilibria.append([E_eq, I_eq])
		eigenvalues.append(eig)
		

	colors = ["cornflowerblue", "coral", "yellowgreen"]

	plt.figure()
	plot_nullclines(p)
	for i, eq in enumerate(equilibria):
		plt.scatter(eq[0], eq[1], marker="x", color=colors[i])
	plot_phase_space(p, t_end, dt)

	plt.figure()
	for i, eig in enumerate(eigenvalues):
		plt.scatter(np.real(eig), np.imag(eig), color=colors[i])
	plt.vlines(0, -1, 1, linestyle="--")
	plt.show()



if __name__ == '__main__':
	# plot_eigenvalues(p)

	plt.figure("Phase space")
	plot_phase_space(p, t_end, dt)
	plot_nullclines(p)
	plt.show()


	# plt.figure(2)
	# x0 = [1., 0.]
	x0 = [0.45, 0.25]
	# plot_simulation(x0, p, t_end, dt)
	# plt.subplot(2, 1, 1)
	# plot_simulation(x0, p, t_end, dt)
	# plt.subplot(2, 1, 2)
	# t = np.linspace(0, t_end, int(t_end/dt))
	# plt.plot(t, input_current(t))
	plt.show()













# param = {'tau_E': 1., 'a_E': 1.2, 'theta_E': 2.8,
# 		 'tau_I': 2., 'a_I': 1.0, 'theta_I': 4.0,
# 		 'wEE': 9, 'wEI': 4, 'wIE': 13, 'wII': 11,
# 		 'I_ext_E': 0, 'I_ext_I': 0}