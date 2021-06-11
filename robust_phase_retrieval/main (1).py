"""
Code for Phase Retrieval Experiments (Section 6.3) of the paper:
Convex-Concave Backtracking for Inertial Bregman Proximal Gradient 
Algorithms in Non-Convex Optimization
Paper authors: Mahesh Chandra Mukkamala, Peter Ochs, 
Thomas Pock and Shoham Sabach.

Other related codes at:
https://github.com/mmahesh/cocain-bpg-escapes-spurious-stationary-points
https://github.com/mmahesh/cocain-bpg-matrix-factorization

Algorithms Implemented:
BPG: Bregman Proximal Gradient
CoCaIn BPG: Convex Concave Inertial (CoCaIn) BPG 
BPG-WB: BPG with Backtracking
IBPM-LS: Inexact Bregman Proximal Minimization Line Search Algorithm

References:
CoCaIn BPG paper: https://arxiv.org/abs/1904.03537
BPG paper: https://arxiv.org/abs/1706.06461
IBPM-LS paper: https://arxiv.org/abs/1707.02278

Contact: Mahesh Chandra Mukkamala (mukkamala@math.uni-sb.de)
"""

import numpy as np

from my_functions import *

import argparse
parser = argparse.ArgumentParser(description='Phase Retrieval Experiments')
parser.add_argument('--lam', '--regularization-parameter',
					default=1e-1, type=float,  dest='lam')
parser.add_argument('--algo', '--algorithm', default=1, type=int,  dest='algo')
parser.add_argument('--max_iter', '--max_iter',
					default=100, type=int,  dest='max_iter')
parser.add_argument('--fun_num', '--fun_num', default=1,
					type=int,  dest='fun_num')
parser.add_argument('--abs_fun_num', '--abs_fun_num',
					default=1, type=int,  dest='abs_fun_num')
parser.add_argument('--breg_num', '--breg_num',
					default=1, type=int,  dest='breg_num')
args = parser.parse_args()


# some backward compatibility and initialization
lam = args.lam
algo = args.algo
fun_num = args.fun_num
abs_fun_num = args.abs_fun_num
breg_num = args.breg_num

# for forward backward splitting only
np.random.seed(0)

max_iter = args.max_iter

dim = 10  # dimension 10
temp_Alist = []
temp_blist = []

global_L = 0

for i in range(100):
	temp_A = np.random.rand(dim, 1)
	A = temp_A*temp_A.T
	temp_Alist = temp_Alist + [A]
	temp_b = np.random.rand(1)[0]
	global_L = global_L + np.amax(np.abs(np.linalg.eigvals(A)))
	temp_blist = temp_blist + [temp_b]

global_L = 2*global_L
print('global_L', global_L)

A = temp_Alist
b = temp_blist

if fun_num == 1:
	# fun_num = 1 for L1 Regularization
	del_val = 0.15
	eps_val = 0.00001
	uL_est =  10
	lL_est = 1e-4*uL_est
	init_U = uL_est
	U = np.ones(dim)
	lam = 1e-1

	prev_U = U


if fun_num == 2:
	# fun_num = 2 for L2 Regularization
	del_val = 0.15
	eps_val = 0.00001
	uL_est = 10
	lL_est = 1e-4*uL_est
	U = np.ones(dim)
	init_U = uL_est
	lam = 1e-1

	prev_U = U

lL_est_main = lL_est

def prox_onto_cube(x):
	return np.maximum(-1.0,np.minimum(1.0,x))

def prox_L1(x, tau):
	return np.maximum(0, np.abs(x)-tau)*np.sign(x)


def prox_squared_L2(x, tau):
	return x/(1+tau)


def make_update(y, grad, uL_est, option=0):
	# option=0 => L1 Regularization
	# option=1 => L2 Regularization
	
	# Bregman Proximal Mapping with L1 Regularization
	# Probably requires PDHG algorithm/ADMM (and depends on its parameters)

	# global variable lam is used.
	
	# y is actually x^k

	# Subproblem objective  
	# TODO: Check once more
	def internal_objective(x,y, tau):	
		# Model around y			
		return abs_func(A, b, x, y, lam, abs_fun_num=abs_fun_num, fun_num=fun_num)\
                    + ((0.5/tau)*breg(x, y, breg_num=breg_num))

	# Constructing K.

	# TODO: Here I am using 100 as default setting make it general
	K = np.zeros((len(y), 100)) # TODO: Should be K^T
	temp_b_2 = np.zeros_like(b) # tilde b
	
	count = 0
	for item in A:
		temp_x_val = 2*item.dot(y)
		K[:, count] = temp_x_val
		temp_b_2[count] = y.T.dot(item.dot(y)) - b[count] - temp_x_val.dot(y)
		count+=1
	
	K = K.T
	# param value
	L_pdhg = np.linalg.norm(K)
	tau = 2000/L_pdhg
	sigma = 0.99/(L_pdhg*2000)

	# primal variables
	x_1 = y.copy()
	x_hat = x_1.copy()

	# dual variables
	p = np.zeros_like(b)

	max_sub_iter = 1000
	for iter in range(max_sub_iter):

		# dual update step
		vec_2 = sigma*K.dot(2*x_1 - x_hat)
		p = prox_onto_cube(p+vec_2+sigma*temp_b_2)
		
		x_hat = x_1.copy()

		if option == 0:
			# primal update step
			x_1 = prox_L1(0.5*(x_1 - tau*K.T.dot(p)) + 0.5*(y), lam*tau)
		else:
			x_1 = prox_squared_L2(0.5*(x_1 - tau*K.T.dot(p)) + 0.5*(y), lam*tau)

		# print('Objective '+ str(internal_objective(x_1,y, tau)) + ' tau ' + str(tau))
		# TODO: Internal objective not giving zero objective when

	return x_1


def find_gamma(A, b, U, prev_U, uL_est, lL_est):
	# Finding the inertial parameter gamma

	gamma = 1
	kappa = (del_val - eps_val)*(uL_est/(uL_est+lL_est))
	y_U = U + gamma*(U-prev_U)
	while (kappa*breg(prev_U, U, breg_num=breg_num, A=A, b=b, lam=lam) \
		< breg(U, y_U, breg_num=breg_num,  A=A, b=b, lam=lam)):
		gamma = gamma*0.9
		y_U = U + gamma*(U-prev_U)
	return y_U, gamma


def find_closed_gamma(A, b, U, prev_U, uL_est, lL_est):
	# Finding the inertial parameter gamma
	# For Euclidean distance
	kappa = (del_val - eps_val)*(uL_est/(uL_est+lL_est))
	gamma = np.sqrt(kappa)
	y_U = U + gamma*(U-prev_U)
	return y_U, gamma


def do_lb_search(A, b, U, U1, lam, uL_est, lL_est, closed_form=0):
	# Lower bound backtracking

	if closed_form==0:
		y_U, gamma = find_gamma(A, b, U, U1, uL_est, lL_est)
	else:
		y_U, gamma = find_closed_gamma(A, b, U, U1, uL_est, lL_est)


	while((abs_func(A, b, U, y_U, lam, abs_fun_num=abs_fun_num, fun_num=fun_num)
			- main_func(A, b, U, lam, fun_num=fun_num)
			- (lL_est*breg(U, y_U,breg_num=breg_num, A=A, b=b, lam=lam))) > 1e-7):

		lL_est = (2)*lL_est
		if closed_form==0:
			y_U, gamma = find_gamma(A, b, U, U1, uL_est, lL_est)
		else:
			y_U, gamma = find_closed_gamma(A, b, U, U1, uL_est, lL_est)

	return lL_est, y_U, gamma


def do_ub_search(A, b, y_U, uL_est):
	# compute gradients
	grad_u = grad(A, b, y_U, lam, fun_num=fun_num)

	# make update step
	if fun_num == 1:
		x_U = make_update(y_U, grad_u, uL_est,option=0)
	elif fun_num == 2:
		x_U = make_update(y_U, grad_u, uL_est, option=1)
	else:
		raise
	
	delta_new = (abs_func(A, b, x_U, y_U, lam, abs_fun_num=abs_fun_num, fun_num=fun_num)
               - main_func(A, b, x_U, lam,  fun_num=fun_num)
               + (uL_est*breg(x_U, y_U, breg_num=breg_num, A=A, b=b, lam=lam)))
	# print('Delta is ' + str(delta_new))
	while((delta_new < -1e-7)) :
		
		delta_prev = delta_new
		delta_new = (abs_func(A, b, x_U, y_U, lam, abs_fun_num=abs_fun_num, 	fun_num=fun_num)
                    - main_func(A, b, x_U, lam,  fun_num=fun_num)
                    + (uL_est*breg(x_U, y_U, breg_num=breg_num, A=A, b=b, lam=lam)))

		# print('Delta is '+ str(delta_new))
		
		uL_est = (2)*uL_est
		print('uL_est is '+ str(uL_est))
		# make update step
		if fun_num == 1:
			x_U = make_update(y_U, grad_u, uL_est, option=0)
		elif fun_num == 2:
			x_U = make_update(y_U, grad_u, uL_est, option=1)
		else:
			raise

	return uL_est, x_U


def obtain_delta(A, b, y_U, uL_est):
	grad_u = grad(A, b, y_U, lam, fun_num=fun_num)
	if fun_num == 1:
		tx_U = make_update(y_U, grad_u, uL_est, option=0)
	elif fun_num == 2:
		tx_U = make_update(y_U, grad_u, uL_est, option=1)
	else:
		raise
	temp_delta = (abs_func(A, b, tx_U, y_U, lam, \
						abs_fun_num=abs_fun_num, fun_num=fun_num)
				  - main_func(A, b, y_U, lam,  fun_num=fun_num)
				  + (uL_est*breg(tx_U, y_U, breg_num=breg_num)))
	return temp_delta, tx_U


def line_search(y_U):
	# TODO: Check the values of gamma and tau
	gm = 0.1
	eta = 0.1
	# here some gm, eta values can be unstable towards the end
	# requires some tuning
	# the above values work fine

	delta, tx_U = obtain_delta(A, b, y_U, uL_est)
	x_U = y_U + eta*(tx_U - y_U)

	while(main_func(A, b, x_U, lam,  fun_num=fun_num) \
		- main_func(A, b, y_U, lam,  fun_num=fun_num) \
		- (eta*gm*delta) > 1e-7) and (delta > 0):
		eta = eta*0.1
		x_U = y_U + eta*(tx_U - y_U)

	return x_U


if algo == 1:
	# Implementation of CoCaIn BPG
	gamma_vals = [0]
	uL_est_vals = [uL_est]
	lL_est_vals = [lL_est]

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	print('temp is ' + str(temp))

	func_vals = [temp]
	lyapunov_vals = [temp]

	U_vals = [init_U]
	# U2_vals = []
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0
	for i in range(max_iter):
		st_time = time.time()
		lL_est, y_U, gamma = do_lb_search(
			A, b, U, prev_U,  lam, uL_est, lL_est=lL_est_main)
		prev_U = U
		# print("doing Lb " + str(lL_est))
		# print("doing Ub " + str(uL_est))
		temp_ulest = uL_est

		uL_est, U = do_ub_search(A, b, y_U, uL_est)
		# print('funct value at '+ str(i) + ' is ')
		print(main_func(A, b, U, lam, fun_num=fun_num))
		uL_est_vals = uL_est_vals + [uL_est]
		lL_est_vals = lL_est_vals + [lL_est]

		gamma_vals = gamma_vals + [gamma]
		U_vals = U_vals + [U]
		temp = main_func(A, b, U, lam, fun_num=fun_num)
		if np.isnan(temp):
			raise
		func_vals = func_vals + [temp]
		lyapunov_vals = lyapunov_vals + \
			[(1/uL_est)*temp+breg(U, prev_U, breg_num=breg_num, A=A, b=b, lam=lam)]
		time_vals[i+1] = time.time() - st_time

	filename = 'results/cocain_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals, lyapunov_vals,
							   uL_est_vals, lL_est_vals, gamma_vals, time_vals])
if algo == 2:
	# Implementation of BPG with Backtracking
	uL_est_vals = [uL_est]

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	prev_fun_val = temp
	func_vals = [temp]
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0

	for i in range(max_iter):
		st_time = time.time()

		uL_est, U = do_ub_search(A, b, U, uL_est)
		print(main_func(A, b, U, lam, fun_num=fun_num))
		print('uL_est is '+ str(uL_est))
		uL_est_vals = uL_est_vals + [uL_est]

		temp = main_func(A, b, U, lam, fun_num=fun_num)
		#print('fun val is '+ str(temp))
		if np.isnan(temp):
			raise
		if (temp >prev_fun_val) and (np.abs(temp-prev_fun_val) > 1e-6):
			print('Function value increases')
			raise
		prev_fun_val = temp
		func_vals = func_vals + [temp]
		time_vals[i+1] = time.time() - st_time

	filename = 'results/gd_bt_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals,  time_vals, uL_est_vals])
if algo == 3:
	# Implementation of BPG without backtracking
	# Here global_L governs the step-size

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	print('temp is ' + str(temp))

	func_vals = [temp]
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0
	for i in range(max_iter):
		st_time = time.time()
		gamma = 0

		uL_est = global_L
		grad_u = grad(A, b, U, lam, fun_num=fun_num)
		if fun_num == 1:
			U = make_update(U, grad_u, uL_est, option=0)
		elif fun_num == 2:
			U = make_update(U, grad_u, uL_est, option=1)
		else:
			raise

		temp = main_func(A, b, U, lam, fun_num=fun_num)
		print('fun val is '+ str(temp))
		if np.isnan(temp):
			raise
		
		func_vals = func_vals + [temp]
		time_vals[i+1] = time.time() - st_time
	filename = 'results/gd_bt_global_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals, time_vals])
if algo == 4:
	# IBPM-LS from https://arxiv.org/abs/1707.02278

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	print('temp is ' + str(temp))

	func_vals = [temp]
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0
	for i in range(max_iter):
		st_time = time.time()
		gamma = 0
		U = line_search(U)
		print(main_func(A, b, U, lam, fun_num=fun_num))
		temp = main_func(A, b, U, lam, fun_num=fun_num)
		if np.isnan(temp):
			raise
		func_vals = func_vals + [temp]
		time_vals[i+1] = time.time() - st_time

	filename = 'results/ibgm_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals, time_vals])

if algo == 5:
	# Implementation of CoCaIn BPG with Closed form Inertia
	gamma_vals = [0]
	uL_est_vals = [uL_est]
	lL_est_vals = [lL_est]

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	print('temp is ' + str(temp))

	func_vals = [temp]
	lyapunov_vals = [temp]

	U_vals = [init_U]

	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0
	for i in range(max_iter):
		st_time = time.time()
		lL_est, y_U, gamma = do_lb_search(
			A, b, U, prev_U,  lam, uL_est, lL_est=lL_est_main, closed_form=1)
		prev_U = U
		# print("doing Lb " + str(lL_est))
		# print("doing Ub " + str(uL_est))
		temp_ulest = uL_est

		uL_est, U = do_ub_search(A, b, y_U, uL_est)
		# print('funct value at '+ str(i) + ' is ')
		print(main_func(A, b, U, lam, fun_num=fun_num))
		uL_est_vals = uL_est_vals + [uL_est]
		lL_est_vals = lL_est_vals + [lL_est]

		gamma_vals = gamma_vals + [gamma]
		U_vals = U_vals + [U]
		temp = main_func(A, b, U, lam, fun_num=fun_num)
		if np.isnan(temp):
			raise
		func_vals = func_vals + [temp]
		lyapunov_vals = lyapunov_vals + \
			[(1/uL_est)*temp+breg(U, prev_U, breg_num=breg_num, A=A, b=b, lam=lam)]
		time_vals[i+1] = time.time() - st_time

	filename = 'results/cocain_cf_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals, lyapunov_vals,
							   uL_est_vals, lL_est_vals, gamma_vals, time_vals])

if algo == 6:
	# Implementation of ALBUM 


	## Primal Step

	## Multiplier Step

	## Adaptive Step-size

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	print('temp is ' + str(temp))

	func_vals = [temp]
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0
	for i in range(max_iter):
		st_time = time.time()
		gamma = 0

		uL_est = global_L
		grad_u = grad(A, b, U, lam, fun_num=fun_num)
		if fun_num == 1:
			U = make_update(U, grad_u, uL_est, option=0)
		elif fun_num == 2:
			U = make_update(U, grad_u, uL_est, option=1)
		else:
			raise

		temp = main_func(A, b, U, lam, fun_num=fun_num)
		print('fun val is ' + str(temp))
		if np.isnan(temp):
			raise

		func_vals = func_vals + [temp]
		time_vals[i+1] = time.time() - st_time
	filename = 'results/gd_bt_global_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals, time_vals])
