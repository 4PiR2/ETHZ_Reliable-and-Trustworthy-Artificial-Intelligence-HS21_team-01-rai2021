import torch


def normalize(x):
	return (x - .1307) / .3081


def get_homogenous_weight(w, b):
	# | w b |
	# | 0 1 |
	return torch.cat([torch.cat([w, torch.zeros((1, w.shape[1]), dtype=torch.float64)], 0),
	                  torch.unsqueeze(torch.cat([b, torch.ones(1, dtype=torch.float64)], 0), 1)], 1)


def get_spu_weight(l, u, f):
	(w_l, b_l), (w_u, b_u) = f(l.flatten()[:-1], u.flatten()[:-1])
	W_l = get_homogenous_weight(torch.diag(w_l), b_l)
	W_u = get_homogenous_weight(torch.diag(w_u), b_u)
	return torch.stack([W_l, W_u])


def back_substitution(eq, dp_eqs, dp_l, dp_u):
	n_layers = len(dp_eqs)
	n_nodes = eq.shape[1]
	dp_eqs.append(eq)
	eq_l = eq[0]
	eq_u = eq[1]
	for i in range(n_layers)[::-1]:
		eq_i = dp_eqs[i]
		eq_l_new = torch.zeros((n_nodes, dp_eqs[i].shape[2]), dtype=torch.float64)
		eq_u_new = torch.zeros((n_nodes, dp_eqs[i].shape[2]), dtype=torch.float64)
		for j in range(n_nodes):
			row = torch.unsqueeze(eq_l[j], 0)
			eq_l_new[j] = row @ torch.where(row.T >= 0., eq_i[0], eq_i[1])
			row = torch.unsqueeze(eq_u[j], 0)
			eq_u_new[j] = row @ torch.where(row.T >= 0., eq_i[1], eq_i[0])
		eq_l = eq_l_new
		eq_u = eq_u_new
	dp_l.append(eq_l)
	dp_u.append(eq_u)
	assert torch.all(eq_l <= eq_u)


def analyze_f(net, inputs, eps, true_label, f):
	params = net.state_dict()
	params = [params[k].double() for k in params]
	params = [get_homogenous_weight(params[2 * i], params[2 * i + 1]) for i in range(len(params) // 2)]

	# eps after normalization?
	inputs = normalize(inputs.flatten().double())
	dp_l = [torch.unsqueeze(torch.cat([inputs - eps, torch.ones(1, dtype=torch.float64)], 0), 1)]
	dp_u = [torch.unsqueeze(torch.cat([inputs + eps, torch.ones(1, dtype=torch.float64)], 0), 1)]
	dp_eqs = [torch.stack([dp_l[-1], dp_u[-1]])]

	for i in range(len(params)):
		back_substitution(torch.stack([params[i]] * 2), dp_eqs, dp_l, dp_u)
		if i < len(params) - 1:
			back_substitution(get_spu_weight(dp_l[-1], dp_u[-1], f), dp_eqs, dp_l, dp_u)

	out_weight = torch.eye(11, dtype=torch.float64)
	out_weight[:, true_label] -= 1.
	out_weight = torch.cat([out_weight[:true_label], out_weight[true_label + 1:-1]], 0)
	back_substitution(torch.stack([out_weight] * 2), dp_eqs, dp_l, dp_u)

	result = torch.all(dp_u[-1] < 0.)
	return result


if __name__ == '__main__':
	# test
	def test_hw6():
		def compute_linear_bounds_test(l, u):
			w_l = torch.tensor([1., 0.], dtype=torch.float64)
			b_l = torch.tensor([0., 0.], dtype=torch.float64)
			w_u = torch.tensor([1, 1. / 3.], dtype=torch.float64)
			b_u = torch.tensor([0, 2. / 3.], dtype=torch.float64)
			return (w_l, b_l), (w_u, b_u)

		w1 = torch.tensor([[1., 1.],
		                   [1., -2.]], dtype=torch.float64)
		b1 = torch.tensor([0., 0.], dtype=torch.float64)
		w2 = torch.tensor([[1., 1.],
		                   [-1., 1.]], dtype=torch.float64)
		b2 = torch.tensor([0., 0.], dtype=torch.float64)

		out_weight = torch.tensor([[1., -1., 0.]], dtype=torch.float64)

		eq1 = torch.stack([get_homogenous_weight(w1, b1)] * 2)
		eq2 = torch.stack([get_homogenous_weight(w2, b2)] * 2)
		eqo = torch.stack([out_weight] * 2)

		dp_l = [torch.unsqueeze(
			torch.cat([torch.tensor([0., 0.], dtype=torch.float64), torch.ones(1, dtype=torch.float64)], 0), 1)]
		dp_u = [torch.unsqueeze(
			torch.cat([torch.tensor([1., 1.], dtype=torch.float64), torch.ones(1, dtype=torch.float64)], 0), 1)]
		dp_eqs = [torch.stack([dp_l[-1], dp_u[-1]])]

		back_substitution(eq1, dp_eqs, dp_l, dp_u)
		back_substitution(get_spu_weight(dp_l[-1], dp_u[-1], compute_linear_bounds_test), dp_eqs, dp_l, dp_u)
		back_substitution(eq2, dp_eqs, dp_l, dp_u)
		back_substitution(eqo, dp_eqs, dp_l, dp_u)

		result = torch.all(dp_u[-1] > 0.)
		return result


	def test_lec6():
		def compute_linear_bounds_test1(l, u):
			w_l = torch.tensor([0., 0.], dtype=torch.float64)
			b_l = torch.tensor([0., 0.], dtype=torch.float64)
			w_u = torch.tensor([.5, .5], dtype=torch.float64)
			b_u = torch.tensor([1., 1.], dtype=torch.float64)
			return (w_l, b_l), (w_u, b_u)

		def compute_linear_bounds_test2(l, u):
			w_l = torch.tensor([0., 0.], dtype=torch.float64)
			b_l = torch.tensor([0., 0.], dtype=torch.float64)
			w_u = torch.tensor([5. / 6., .5], dtype=torch.float64)
			b_u = torch.tensor([5. / 12., 1.], dtype=torch.float64)
			return (w_l, b_l), (w_u, b_u)

		w1 = torch.tensor([[1., 1.],
		                   [1., -1.]], dtype=torch.float64)
		b1 = torch.tensor([0., 0.], dtype=torch.float64)
		w2 = torch.tensor([[1., 1.],
		                   [1., -1.]], dtype=torch.float64)
		b2 = torch.tensor([-.5, 0.], dtype=torch.float64)
		w3 = torch.tensor([[-1., 1.],
		                   [0., 1.]], dtype=torch.float64)
		b3 = torch.tensor([3., 0.], dtype=torch.float64)

		out_weight = torch.tensor([[1., -1., 0]], dtype=torch.float64)

		eq1 = torch.stack([get_homogenous_weight(w1, b1)] * 2)
		eq2 = torch.stack([get_homogenous_weight(w2, b2)] * 2)
		eq3 = torch.stack([get_homogenous_weight(w3, b3)] * 2)
		eqo = torch.stack([out_weight] * 2)

		dp_l = [torch.unsqueeze(
			torch.cat([torch.tensor([-1., -1.], dtype=torch.float64), torch.ones(1, dtype=torch.float64)], 0), 1)]
		dp_u = [torch.unsqueeze(
			torch.cat([torch.tensor([1., 1.], dtype=torch.float64), torch.ones(1, dtype=torch.float64)], 0), 1)]
		dp_eqs = [torch.stack([dp_l[-1], dp_u[-1]])]

		back_substitution(eq1, dp_eqs, dp_l, dp_u)
		back_substitution(get_spu_weight(dp_l[-1], dp_u[-1], compute_linear_bounds_test1), dp_eqs, dp_l, dp_u)
		back_substitution(eq2, dp_eqs, dp_l, dp_u)
		back_substitution(get_spu_weight(dp_l[-1], dp_u[-1], compute_linear_bounds_test2), dp_eqs, dp_l, dp_u)
		back_substitution(eq3, dp_eqs, dp_l, dp_u)
		back_substitution(eqo, dp_eqs, dp_l, dp_u)

		result = torch.all(dp_u[-1] > 0.)
		return result


	print(test_hw6())
	print(test_lec6())
	pass
