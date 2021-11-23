import torch


def get_const(x, shape=1, device='cpu', ref=None):
	# get a tensor of constant numbers
	if ref is not None:
		shape = ref.shape
		device = ref.device
	if x == 0.:
		return torch.zeros(shape, dtype=torch.float64, device=device)
	else:
		return x * torch.ones(shape, dtype=torch.float64, device=device)

def spu_0(x):
	# spu function (optimized)
	y = get_const(0., ref=x)
	mask = x >= 0.
	y[mask] = x[mask] ** 2 - .5
	mask = ~mask
	y[mask] = -torch.sigmoid(x[mask])
	return y

def spu_1(x):
	# 1-order derivative
	# spu'(0) := 0
	y = get_const(0., ref=x)
	mask = x >= 0.
	y[mask] = 2. * x[mask]
	mask = ~mask
	f = lambda s: s * (s - 1.)
	y[mask] = f(torch.sigmoid(x[mask]))
	return y

def get_tanget(t):
	w = spu_1(t)
	b = spu_0(t) - w * t
	return w, b

def get_scant(a, b):
	ya, yb = spu_0(a), spu_0(b)
	w = (yb - ya) / (b - a)
	b = ya - w * a
	return w, b

def case_0ltu(l, u, t=None):
	# case 0 <= l <= t <= u
	# L: tangent @ t
	# U: scant @ l, u
	w_u = l + u
	b_u = - l * u - .5
	if t is None:
		# t = (l + u) * .5
		w_l = w_u
		b_l = -.25 * w_u ** 2 - .5
	else:
		w_l = 2. * t
		b_l = - t ** 2 - .5
	return w_l, b_l, w_u, b_u

def case_ltu0(l, u, t=None):
	# case l <= t <= u <= 0
	# L: scant @ l, u
	# U: tangent @ t
	s_l = torch.sigmoid(l)
	s_u = torch.sigmoid(u)
	u_sub_l = u - l
	w_l = (s_l - s_u) / u_sub_l
	b_l = (l * s_u - u * s_l) / u_sub_l
	if t is None:
		t = (l + u) * .5
	s_t = torch.sigmoid(t)
	w_u = s_t * (s_t - 1.)
	b_u = - w_u * t - s_t
	return w_l, b_l, w_u, b_u

def case_l0tu(l, u, t):
	# case l < 0 <= t <= u
	# L: tangent @ t
	# U1: scant @ l, u
	# U2: tangent @ l
	w_l = 2. * t
	b_l = - t ** 2 - .5
	s_l = torch.sigmoid(l)
	y_u = u ** 2 - .5
	u_sub_l = u - l
	w_u = (y_u + s_l) / u_sub_l
	w_u_2 = s_l * (s_l - 1.)
	mask = w_u >= w_u_2
	b_u = get_const(0., ref=l)
	b_u[mask] = - (l[mask] * y_u[mask] + u[mask] * s_l[mask]) / u_sub_l[mask]
	mask = ~mask
	w_u[mask] = w_u_2[mask]
	b_u[mask] = - w_u_2[mask] * l[mask] - s_l[mask]
	return w_l, b_l, w_u, b_u

def case_lt0u(l, u, t):
	# case l <= t < 0 < u
	# L: scant @ l, 0
	# U1: tangent @ t
	# U2: scant @ l, u
	# U3: tangent @ l
	# U4: tangent @ a, solve s(a)(s(a) - 1.) = (u ** 2 - .5 + s(a)) / (u - a), not implemented
	# u++ ==> U1 -> U4 -> U3 -> U2
	s_l = torch.sigmoid(l)
	w_l = (.5 - s_l) / l
	b_l = get_const(-.5, ref=l)
	s_t = torch.sigmoid(t)
	w_u = s_t * (s_t - 1.)
	b_u = - w_u * t - s_t
	# test @ u
	mask1 = u ** 2 - .5 > w_u * u + b_u
	y_u = u[mask1] ** 2 - .5
	u_sub_l = u[mask1] - l[mask1]
	w_u_2 = (y_u + s_l[mask1]) / u_sub_l
	w_u_3 = s_l[mask1] * (s_l[mask1] - 1.)
	# compare slope
	mask2 = w_u_2 >= w_u_3
	b_u_2 = get_const(0., ref=w_u_2)
	b_u_2[mask2] = - (l[mask1][mask2] * y_u[mask2] + u[mask1][mask2] * s_l[mask1][mask2]) / u_sub_l[mask2]
	mask2 = ~mask2
	w_u_2[mask2] = w_u_3[mask2]
	b_u_2[mask2] = - w_u_3[mask2] * l[mask1][mask2] - s_l[mask1][mask2]
	w_u[mask1] = w_u_2
	b_u[mask1] = b_u_2
	return w_l, b_l, w_u, b_u

def case_l0u(l, u, t=None):
	# case l < 0 < u
	if t is None:
		t = (l + u) * .5
	w_l = get_const(0., ref=l)
	b_l = get_const(0., ref=l)
	w_u = get_const(0., ref=l)
	b_u = get_const(0., ref=l)
	mask = t >= 0.
	w_l[mask], b_l[mask], w_u[mask], b_u[mask] = case_l0tu(l[mask], u[mask], t[mask])
	mask = ~mask
	w_l[mask], b_l[mask], w_u[mask], b_u[mask] = case_lt0u(l[mask], u[mask], t[mask])
	return w_l, b_l, w_u, b_u

def compute_linear_bounds(l, u):
	# input: n-dim vector l, u (float64 tensor)
	# output: n-dim vector w_l, b_l, w_u, b_u (float64 tensor)
	w_l = get_const(0., ref=l)
	b_l = get_const(0., ref=l)
	w_u = get_const(0., ref=l)
	b_u = get_const(0., ref=l)
	mask1 = l >= 0.
	w_l[mask1], b_l[mask1], w_u[mask1], b_u[mask1] = case_0ltu(l[mask1], u[mask1])
	mask2 = u <= 0.
	w_l[mask2], b_l[mask2], w_u[mask2], b_u[mask2] = case_ltu0(l[mask2], u[mask2])
	mask3 = ~torch.bitwise_or(mask1, mask2)
	w_l[mask3], b_l[mask3], w_u[mask3], b_u[mask3] = case_l0u(l[mask3], u[mask3])
	return (w_l, b_l), (w_u, b_u)


if __name__ == '__main__':
	# test
	epsilon = 1e-10
	n = 10000
	visualize = 100

	l, u = torch.randn(n, dtype=torch.float64), torch.randn(n, dtype=torch.float64)
	l, u = torch.minimum(l, u), torch.maximum(l, u)
	(w_l, b_l), (w_u, b_u) = compute_linear_bounds(l, u)
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
			plt.savefig('../tmp/' + str(i) + '.jpg')
			# plt.show()
			plt.clf()
