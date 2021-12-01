import torch


def get_const_tensor(x, shape=1):
	# get a tensor of constant numbers
	return x * torch.ones(shape, dtype=torch.float64)


def spu_0(x):
	# spu function (optimized)
	y = get_const_tensor(0., x.shape)
	mask = x >= 0.
	y[mask] = x[mask] ** 2 - .5
	mask = ~mask
	y[mask] = -torch.sigmoid(x[mask])
	return y


def spu_1(x):
	# 1-order derivative
	# spu'(0) := 0
	y = get_const_tensor(0., x.shape)
	mask = x >= 0.
	y[mask] = 2. * x[mask]
	mask = ~mask
	f = lambda s: s * (s - 1.)
	y[mask] = f(torch.sigmoid(x[mask]))
	return y


def get_tanget_line(t):
	w = spu_1(t)
	b = spu_0(t) - w * t
	return w, b


def get_scant_line(p, q):
	m, n = spu_0(p), spu_0(q)
	q_sub_p = q - p
	w = (n - m) / q_sub_p
	b = (m * q - n * p) / q_sub_p  # m - w * p
	mask = p == q
	w[mask], b[mask] = get_tanget_line(p[mask])
	return w, b


def solve_U2(X, U):
	# Newton's method
	t = get_const_tensor(0., len(X))
	for i in range(len(X)):
		x, u = X[i], U[i]
		u2 = u ** 2
		a = u2 - .5
		b = 2. * u2 + u
		c = u2 + .5

		ex = torch.exp(-x)
		aex2 = a * ex ** 2
		x_b = x - b
		f0 = aex2 - x_b * ex + c
		f1 = -2. * aex2 + x_b - 1.
		f0_pre = f0 - 1.
		while f0_pre < f0:
			f0_pre = f0
			x -= f0 / f1
			ex = torch.exp(-x)
			aex2 = a * ex ** 2
			x_b = x - b
			f0 = aex2 - x_b * ex + c
			f1 = -2. * aex2 + x_b - 1.
		t[i] = x
	return t


def lb_base(l, u, t_l=.5, t_u=.5):
	# compute linear bounds base function
	# convex hull tangent line model
	# default: tangent @ mid-point (minimum area trapezoid)
	# input: n-dim vector l, u (float64 tensor)
	# output: n-dim vector w_l, b_l, w_u, b_u (float64 tensor)
	if type(t_l) is float:
		t_l = (1. - t_l) * l + t_l * u
	if type(t_u) is float:
		t_u = (1. - t_u) * l + t_u * u
	# L1: scant @ l, min(0, u) (t < 0)
	# L2: tangent @ t (t >= 0)
	w_l, b_l = get_tanget_line(t_l)
	# test @ 0
	mask = b_l > -.5
	w_l[mask], b_l[mask] = get_scant_line(l[mask], torch.minimum(get_const_tensor(0., len(l[mask])), u[mask]))
	# U1: tangent @ t
	# U2: tangent @ a (solve s(a)(s(a) - 1.) = (u ** 2 - .5 + s(a)) / (u - a))
	# U3: tangent @ l (never used for u >= -l (mask3 always false))
	# U4: scant @ l, u
	tolerance = 1e-15  # lazy switch
	w_u, b_u = get_tanget_line(t_u)
	# test @ u
	mask3 = torch.bitwise_or(w_u * u + b_u + tolerance < spu_0(u), t_u >= 0.)
	w_u[mask3], b_u[mask3] = get_tanget_line(l[mask3])
	# test @ u
	mask4 = w_u * u + b_u + tolerance < spu_0(u)
	w_u[mask4], b_u[mask4] = get_scant_line(l[mask4], u[mask4])
	mask2 = torch.bitwise_and(mask3, ~mask4)
	w_u[mask2], b_u[mask2] = get_tanget_line(solve_U2(l[mask2], u[mask2]))
	return (w_l, b_l), (w_u, b_u)


def lb_boxlike(l, u):
	# optimal box-comparable
	return lb_base(l, u, torch.minimum(torch.maximum(get_const_tensor(0., len(l)), l), u), l)


def lb_parallelogram(l, u):
	# minimum area parallelogram
	(w_l, b_l), (w_u, b_u) = lb_base(l, u)
	mask1 = w_u >= 0.
	w_l[mask1], b_l[mask1] = get_tanget_line(w_u[mask1] * .5)
	mask2 = torch.bitwise_or(torch.bitwise_and(u > 0., w_l < w_u), torch.bitwise_and(w_l > 0., w_u < 0.))
	w_l[mask2] = w_u[mask2]
	b_l[mask2] = -.5
	mask3 = w_l != w_u
	t = torch.sqrt(1. + 4. * w_l[mask3])
	t = torch.log(1. - t) - torch.log(1. + t)
	_, b_u[mask3] = get_tanget_line(t)
	return (w_l, b_l), (w_l, b_u)


def lb_box(l, u):
	# box, not recommended
	w_l = get_const_tensor(0., len(l))
	w_u = w_l
	y_l, y_u = spu_0(l), spu_0(u)
	b_l = get_const_tensor(0., len(l))
	b_u = get_const_tensor(0., len(l))
	mask1 = l >= 0.
	b_l[mask1], b_u[mask1] = y_l[mask1], y_u[mask1]
	mask2 = u <= 0.
	b_l[mask2], b_u[mask2] = y_u[mask2], y_l[mask2]
	mask3 = ~torch.bitwise_or(mask1, mask2)
	b_l[mask3], b_u[mask3] = get_const_tensor(-.5, len(l[mask3])), torch.maximum(y_l[mask3], y_u[mask3])
	return (w_l, b_l), (w_u, b_u)


def lb_random_mix(l, u, f_list):
	# random selection for each node
	index = torch.randint(len(f_list), [len(l)])
	w_l = get_const_tensor(0., len(l))
	b_l = get_const_tensor(0., len(l))
	w_u = get_const_tensor(0., len(l))
	b_u = get_const_tensor(0., len(l))
	for i in range(len(f_list)):
		mask = index == i
		(w_l[mask], b_l[mask]), (w_u[mask], b_u[mask]) = f_list[i](l[mask], u[mask])
	return (w_l, b_l), (w_u, b_u)


if __name__ == '__main__':
	# test
	f = lambda l, u: lb_random_mix(l, u, [lb_base, lb_boxlike, lb_parallelogram, lb_little, lb_box])
	n = 1000
	visualize = 0
	epsilon = 1e-10

	l, u = torch.randn(n, dtype=torch.float64), torch.randn(n, dtype=torch.float64)
	l, u = torch.minimum(l, u), torch.maximum(l, u)
	(w_l, b_l), (w_u, b_u) = f(l, u)
	x = torch.stack([torch.linspace(li, ui, 10000, dtype=torch.float64) for li, ui in zip(l, u)], dim=1)
	y = spu_0(x)
	diff_l = w_l * x + b_l - y
	diff_u = w_u * x + b_u - y
	assert torch.all(diff_l <= epsilon)
	assert torch.all(diff_u >= -epsilon)

	if visualize > 0:
		import matplotlib.pyplot as plt

		for i in range(visualize):
			plt.plot(x[:, i], y[:, i], c='k')
			plt.plot([l[i]] * 2, [w_l[i] * l[i] + b_l[i], w_u[i] * l[i] + b_u[i]], c='g')
			plt.plot([u[i]] * 2, [w_l[i] * u[i] + b_l[i], w_u[i] * u[i] + b_u[i]], c='g')
			plt.plot(x[:, i], w_l[i] * x[:, i] + b_l[i], c='b')
			plt.plot(x[:, i], w_u[i] * x[:, i] + b_u[i], c='r')
			plt.grid()
			plt.title(str(i))
			plt.savefig('../tmp/lb_plots/' + str(i) + '.jpg')
			# plt.show()
			plt.clf()
