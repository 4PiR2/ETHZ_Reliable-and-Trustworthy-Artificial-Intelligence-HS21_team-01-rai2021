import torch

from lb import get_const_tensor, spu_0


def spu_1_inv(x):
	# inverse function of 1-order derivative
	# x < -.25: return nan
	y = get_const_tensor(0., len(x))
	mask = x >= 0.
	y[mask] = x[mask] * .5
	mask = ~mask
	sqrt_delta = torch.sqrt(1. + 4. * x[mask])
	y[mask] = torch.log(1. - sqrt_delta) - torch.log(1. + sqrt_delta)
	return y


def get_intercept(x, w):
	return spu_0(x) - w * x


def get_intercept_choices(l, u, w):
	# line @ l, u, t, 0
	bl = get_intercept(l, w)
	bu = get_intercept(u, w)
	b0 = get_const_tensor(torch.nan, len(l))
	b0[torch.bitwise_and(l < 0., u > 0.)] = -.5
	t = spu_1_inv(w)
	mask = torch.bitwise_and(l < t, u > t)
	bt = get_const_tensor(torch.nan, len(l))
	bt[mask] = get_intercept(t[mask], w[mask])
	return bl, bu, bt, b0


def lb_slope(l, u, w_l, w_u):
	b_l = get_const_tensor(torch.inf, len(l))
	for v in get_intercept_choices(l, u, w_l):
		mask = v < b_l
		b_l[mask] = v[mask]
	b_u = get_const_tensor(-torch.inf, len(l))
	for v in get_intercept_choices(l, u, w_u)[:-1]:
		mask = v > b_u
		b_u[mask] = v[mask]
	return b_l, b_u


def is_intersect(l, u, w_1, b_1, w_2, b_2):
	# w_1 == w_2 and b_1 == b_2: return false
	x = (b_2 - b_1) / (w_1 - w_2)
	return torch.bitwise_and(l < x, u > x)


if __name__ == '__main__':
	# test
	n = 1000
	visualize = 100
	epsilon = 1e-10

	w_l, w_u = torch.randn(n, dtype=torch.float64), torch.randn(n, dtype=torch.float64)
	l, u = torch.randn(n, dtype=torch.float64), torch.randn(n, dtype=torch.float64)
	l, u = torch.minimum(l, u), torch.maximum(l, u)
	b_l, b_u = lb2(l, u, w_l, w_u)
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
