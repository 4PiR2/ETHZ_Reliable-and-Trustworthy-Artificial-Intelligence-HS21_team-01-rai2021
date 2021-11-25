import torch


def get_const_tensor(x, shape=1, ref=None):
	# get a tensor of constant numbers
	if ref is not None:
		shape = ref.shape
	if x == 0.:
		return torch.zeros(shape, dtype=torch.float64)
	else:
		return x * torch.ones(shape, dtype=torch.float64)


def spu_0(x):
	# spu function (optimized)
	y = get_const_tensor(0., ref=x)
	mask = x >= 0.
	y[mask] = x[mask] ** 2 - .5
	mask = ~mask
	y[mask] = -torch.sigmoid(x[mask])
	return y


def spu_1(x):
	# 1-order derivative
	# spu'(0) := 0
	y = get_const_tensor(0., ref=x)
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
	return w, b


def case_0lu(l, u, t_l=None):
	# case 0 <= l <= u
	# L: tangent @ t
	# U: scant @ l, u
	w_u = l + u
	b_u = - l * u - .5
	if t_l is None:
		# t = (l + u) * .5
		w_l = w_u
		b_l = -.25 * w_u ** 2 - .5
	else:
		w_l = 2. * t_l
		b_l = - t_l ** 2 - .5
	return w_l, b_l, w_u, b_u


def case_lu0(l, u, t_u=None):
	# case l <= u <= 0
	# L: scant @ l, u
	# U: tangent @ t
	s_l = torch.sigmoid(l)
	s_u = torch.sigmoid(u)
	u_sub_l = u - l
	w_l = (s_l - s_u) / u_sub_l
	b_l = (l * s_u - u * s_l) / u_sub_l
	if t_u is None:
		t_u = (l + u) * .5
	s_t = torch.sigmoid(t_u)
	w_u = s_t * (s_t - 1.)
	b_u = - w_u * t_u - s_t
	return w_l, b_l, w_u, b_u


def case_l0u_l(l, t_l):
	# case l < 0 < u
	# L1: tangent @ t (t >= 0)
	# L2: scant @ l, 0 (t < 0)
	w_l = get_const_tensor(0., ref=l)
	b_l = get_const_tensor(0., ref=l)
	mask = t_l >= 0.
	w_l[mask] = 2. * t_l[mask]
	b_l[mask] = - t_l[mask] ** 2 - .5
	mask = ~mask
	s_l = torch.sigmoid(l[mask])
	w_l[mask] = (.5 - s_l) / l[mask]
	b_l[mask] = get_const_tensor(-.5, ref=s_l)
	return w_l, b_l


def case_l0u_u(l, u, t_u):
	# case l <= t < 0 < u
	# U1: tangent @ t
	# U2: scant @ l, u
	# U3: tangent @ l (never used for u >= -l (mask always false))
	# U4: tangent @ a (solve s(a)(s(a) - 1.) = (u ** 2 - .5 + s(a)) / (u - a), not implemented)
	# u++ ==> U1 -> U4 -> U3 -> U2
	s_t = torch.sigmoid(t_u)
	w_u = s_t * (s_t - 1.)
	b_u = - w_u * t_u - s_t
	# test @ u
	mask1 = u ** 2 - .5 > w_u * u + b_u
	y_u = u[mask1] ** 2 - .5
	s_l = torch.sigmoid(l[mask1])
	u_sub_l = u[mask1] - l[mask1]
	w_u_2 = (y_u + s_l) / u_sub_l
	w_u_3 = s_l * (s_l - 1.)
	# compare slope
	mask2 = w_u_2 >= w_u_3
	b_u_2 = get_const_tensor(0., ref=w_u_2)
	b_u_2[mask2] = - (l[mask1][mask2] * y_u[mask2] + u[mask1][mask2] * s_l[mask2]) / u_sub_l[mask2]
	mask2 = ~mask2
	w_u_2[mask2] = w_u_3[mask2]
	b_u_2[mask2] = - w_u_3[mask2] * l[mask1][mask2] - s_l[mask2]
	w_u[mask1] = w_u_2
	b_u[mask1] = b_u_2
	return w_u, b_u


def case_l0u(l, u, t_l=None, t_u=None):
	# case l < 0 < u
	if t_l is None:
		t_l = (l + u) * .5
	if t_u is None:
		t_u = (l + u) * .5
	w_l, b_l = case_l0u_l(l, t_l)
	w_u, b_u = case_l0u_u(l, u, t_u)
	return w_l, b_l, w_u, b_u


def case_lu(l):
	# case l = u
	w = get_const_tensor(0., ref=l)
	b = spu_0(l)
	return w, b, w, b


def compute_linear_bounds(l, u, t_l=.5, t_u=.5):
	# minimum area
	# input: n-dim vector l, u (float64 tensor)
	# output: n-dim vector w_l, b_l, w_u, b_u (float64 tensor)
	w_l = get_const_tensor(0., ref=l)
	b_l = get_const_tensor(0., ref=l)
	w_u = get_const_tensor(0., ref=l)
	b_u = get_const_tensor(0., ref=l)
	if type(t_l) is float:
		t_l = (1. - t_l) * l + t_l * u
	if type(t_u) is float:
		t_u = (1. - t_u) * l + t_u * u
	mask1 = l >= 0.
	w_l[mask1], b_l[mask1], w_u[mask1], b_u[mask1] = case_0lu(l[mask1], u[mask1], t_l[mask1])
	mask2 = u <= 0.
	w_l[mask2], b_l[mask2], w_u[mask2], b_u[mask2] = case_lu0(l[mask2], u[mask2], t_u[mask2])
	mask3 = ~torch.bitwise_or(mask1, mask2)
	w_l[mask3], b_l[mask3], w_u[mask3], b_u[mask3] = case_l0u(l[mask3], u[mask3], t_l[mask3], t_u[mask3])
	mask4 = l == u
	w_l[mask4], b_l[mask4], w_u[mask4], b_u[mask4] = case_lu(l[mask4])
	return (w_l, b_l), (w_u, b_u)


def compute_linear_bounds_boxlike(l, u):
	# optimal box-comparable
	return compute_linear_bounds(l, u, torch.maximum(get_const_tensor(0., ref=l), l), l)


def compute_linear_bounds_box(l, u):
	# box, not recommended
	# input: n-dim vector l, u (float64 tensor)
	# output: n-dim vector w_l, b_l, w_u, b_u (float64 tensor)
	w_l = get_const_tensor(0., ref=l)
	w_u = w_l
	y_l, y_u = spu_0(l), spu_0(u)
	b_l = get_const_tensor(0., ref=l)
	b_u = get_const_tensor(0., ref=l)
	mask1 = l >= 0.
	b_l[mask1], b_u[mask1] = y_l[mask1], y_u[mask1]
	mask2 = u <= 0.
	b_l[mask2], b_u[mask2] = y_u[mask2], y_l[mask2]
	mask3 = ~torch.bitwise_or(mask1, mask2)
	b_l[mask3], b_u[mask3] = get_const_tensor(-.5, ref=l[mask3]), torch.maximum(y_l[mask3], y_u[mask3])
	return (w_l, b_l), (w_u, b_u)


if __name__ == '__main__':
	# test
	epsilon = 1e-10
	n = 10000
	visualize = 100

	l, u = torch.randn(n, dtype=torch.float64), torch.randn(n, dtype=torch.float64)
	l, u = torch.minimum(l, u), torch.maximum(l, u)
	(w_l, b_l), (w_u, b_u) = compute_linear_bounds(l, u, .5, .5)
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
