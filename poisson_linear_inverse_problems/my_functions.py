import numpy as np
np.random.seed(0)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.image import imread

# utils file to collect important functions


def main_func(A,b, U, lam, fun_num=1):
	if fun_num==1:
		temp_sum = 0
		count = 0
		for item in A:
			temp_ent = item.dot(U) - (b[count]*np.log(item.dot(U)))
			count +=1
			temp_sum = temp_sum + temp_ent
		temp_sum = temp_sum + lam*(np.sum(np.abs(U))) 

		return temp_sum

	if fun_num==2:
		temp_sum = 0
		count = 0
		for item in A:
			temp_ent = item.dot(U) - (b[count]*np.log(item.dot(U)))
			count +=1
			temp_sum = temp_sum + temp_ent
		temp_sum = temp_sum + lam*(U.T.dot(U))

		return temp_sum

	if fun_num == 3:
		temp_sum = 0
		count = 0
		for item in A:
			temp_ent = item.dot(U) - (b[count]*np.log(item.dot(U)))
			count += 1
			temp_sum = temp_sum + temp_ent
		temp_sum = temp_sum 

		return temp_sum


def grad(A,b, U,  lam, fun_num=1):
	if fun_num in [1,3]:
		temp_grad = 0
		count = 0
		for item in A:
			temp_grad = temp_grad + (1 - (b[count]/(item.dot(U))))*item
			count +=1

		return temp_grad

	if fun_num == 2:
		temp_grad = 0
		count = 0
		for item in A:
			temp_grad = temp_grad + (1 - (b[count]/(item.dot(U))))*item
			count +=1
		return temp_grad 

	

def abs_func(A,b, U, U1, lam, abs_fun_num=1, fun_num=1):
	if abs_fun_num == 1:
		G = grad(A,b, U1, lam, fun_num=fun_num)
		return main_func(A,b, U1, lam, fun_num=fun_num)\
							+ np.sum(np.multiply(U-U1,G)) \
							-lam*(np.sum(np.abs(U1))) + lam*(np.sum(np.abs(U)))
	if abs_fun_num == 2:
		G = grad(A,b, U1, lam, fun_num=fun_num)
		return main_func(A,b, U1, lam, fun_num=fun_num) \
				+ np.sum(np.multiply(U-U1,G))-lam*(U1.T.dot(U1))+lam*(U.T.dot(U))
	if abs_fun_num == 3:
		G = grad(A, b, U1, lam, fun_num=fun_num)
		return main_func(A, b, U1, lam, fun_num=fun_num)\
                    + np.sum(np.multiply(U-U1, G))
	


	
def breg( U, U1, breg_num=1, A=1,b=1, lam=1):
	if breg_num==1:
		grad_U1 = -1/U1
		temp = -np.sum(np.log(U)) + np.sum(np.log(U1))- np.sum(np.multiply(U-U1,grad_U1))
		if temp >=1e-10:
			return temp
		else:
			return 0
