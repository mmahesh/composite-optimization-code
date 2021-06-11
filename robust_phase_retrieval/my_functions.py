import numpy as np
np.random.seed(0)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.image import imread



def main_func(A,b, U, lam, fun_num=1):

	if fun_num==1:
		# Robust Phase Retrieval with L1 Regularization
		temp_sum = 0
		count = 0
		for item in A:
			temp_ent = np.abs(U.T.dot(item.dot(U)) - b[count])
			count +=1
			temp_sum = temp_sum + temp_ent
		
		temp_sum = temp_sum + lam*(np.sum(np.abs(U))) 

		return temp_sum

	if fun_num==2:
		# Robust Phase Retrieval with L1 Regularization
		temp_sum = 0
		count = 0
		for item in A:
			temp_ent = np.abs(U.T.dot(item.dot(U)) - b[count])
			count +=1
			temp_sum = temp_sum + temp_ent
		temp_sum = temp_sum + lam*(U.T.dot(U))

		return temp_sum

def grad(A,b, U,  lam, fun_num=1):
	# Gives internal gradient as a list
	if fun_num==1:
		temp_grad = []
		count = 0
		for item in A:
			temp_grad = temp_grad + [2*(item.dot(U))]
			count +=1

		return temp_grad

	if fun_num == 2:
		temp_grad = []
		count = 0
		for item in A:
			temp_grad = temp_grad + [2*(item.dot(U))]
			count +=1
		return temp_grad 

	
def internal_main_func(A, b, U, U1, G1, lam, fun_num=1):
	# G1 = Grad U1
	if fun_num == 1:
		# Robust Phase Retrieval with L1 Regularization
		temp_sum = 0
		count = 0
		for item in A:
			temp_ent = np.abs(U1.T.dot(item.dot(U1)) - b[count] \
				+ np.sum(np.multiply(U-U1, G1[count])))
			count += 1
			temp_sum = temp_sum + temp_ent
		temp_sum = temp_sum + lam*(np.sum(np.abs(U)))

		return temp_sum

	if fun_num == 2:
		# Robust Phase Retrieval with L1 Regularization
		temp_sum = 0
		count = 0
		for item in A:
			temp_ent=np.abs(U1.T.dot(item.dot(U1)) - b[count] \
                            + np.sum(np.multiply(U-U1, G1[count])))
			count += 1
			temp_sum = temp_sum + temp_ent
		temp_sum = temp_sum + lam*(U.T.dot(U))

		return temp_sum

def abs_func(A,b, U, U1, lam, abs_fun_num=1, fun_num=1):
	if abs_fun_num == 1:
		G = grad(A,b, U1, lam, fun_num=fun_num)
		return internal_main_func(A, b, U, U1, G, lam, fun_num=fun_num) 
	if abs_fun_num == 2:
		# TODO: Check this once again.
		G = grad(A,b, U1, lam, fun_num=fun_num)
		return internal_main_func(A, b, U, U1, G, lam, fun_num=fun_num) 


def breg( U, U1, breg_num=1, A=1,b=1, lam=1):
	if breg_num==1:
		grad_U1 = (np.sum(np.multiply(U1,U1)))*U1 + U1
		temp =  0.25*(np.linalg.norm(U)**4) + 0.5*(np.linalg.norm(U)**2) \
				- 0.25*(np.linalg.norm(U1)**4) - 0.5*(np.linalg.norm(U1)**2)\
				- np.sum(np.multiply(U-U1,grad_U1))
		if temp >=1e-15:
			return temp
		else:
			return 0
	if breg_num == 2:
		return 0.5*(np.linalg.norm(U-U1)**2)
