import numpy as np
import matplotlib.pyplot as plt


def spu(x):
	y = np.zeros(len(x))
	mask = x >= 0.
	y[mask] = x[mask] ** 2 - .5
	mask = ~mask
	y[mask] = -1. / (1. + np.exp(-x[mask]))
	return y


mat = []
with open('../tmp/bs_spu_log/1.txt', 'r') as f:
	for line in f:
		mat.append([float(x) for x in line.strip().split()])
mat = np.array(mat)

for i in range(len(mat)):
	l, u, w_l, b_l, w_u, b_u = mat[i]
	x = np.linspace(l, u, 1000)
	y = spu(x)
	plt.plot(x, y, c='k')
	plt.plot([l] * 2, [w_l * l + b_l, w_u * l + b_u], c='g')
	plt.plot([u] * 2, [w_l * u + b_l, w_u * u + b_u], c='g')
	plt.plot([l, u], [w_l * l + b_l, w_l * u + b_l], c='b')
	plt.plot([l, u], [w_u * l + b_u, w_u * u + b_u], c='r')
	plt.grid()
	plt.title(str(i))
	plt.savefig('../tmp/bs_spu_plots/' + str(i) + '.jpg')
	# plt.show()
	plt.clf()
