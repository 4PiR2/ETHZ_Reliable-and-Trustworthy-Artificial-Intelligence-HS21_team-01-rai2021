import torch


def normalize(x):
	return (x - .1307) / .3081


def get_homogeneous_weight(w, b):
	# | w b |
	# | 0 1 |
	return torch.cat([torch.cat([w, torch.zeros((1, w.shape[-1]), dtype=torch.float64)], 0),
	                  torch.unsqueeze(torch.cat([b, torch.ones(1, dtype=torch.float64)], 0), 1)], 1)


def get_spu_weights(l, u, f):
	(w_l, b_l), (w_u, b_u) = f(l.flatten()[:-1], u.flatten()[:-1])
	W_l = get_homogeneous_weight(torch.diag(w_l), b_l)
	W_u = get_homogeneous_weight(torch.diag(w_u), b_u)
	return W_l, W_u


def add_weights(weights_l, weights_u, is_affine_layers, *args):
	weights_l.append(args[0])
	weights_u.append(args[-1])
	is_affine_layers.append(len(args) < 2)


def back_substitution(weights_l, weights_u, is_affine_layers=None):
	n_layers = len(weights_l)
	l, u = weights_l[-1], weights_u[-1]
	n_nodes = len(l)
	for i in range(n_layers - 1)[::-1]:
		if is_affine_layers is not None and is_affine_layers[i]:
			l @= weights_l[i]
			u @= weights_u[i]
		else:
			l_new = torch.zeros((n_nodes, weights_l[i].shape[-1]), dtype=torch.float64)
			u_new = torch.zeros((n_nodes, weights_u[i].shape[-1]), dtype=torch.float64)
			for j in range(n_nodes):
				row = torch.unsqueeze(l[j], 0)
				l_new[j] = row @ torch.where(row.T >= 0., weights_l[i], weights_u[i])
				row = torch.unsqueeze(u[j], 0)
				u_new[j] = row @ torch.where(row.T >= 0., weights_u[i], weights_l[i])
			l, u = l_new, u_new
	# assert torch.all(l <= u)
	return l, u


def analyze_f(net, inputs, eps, true_label, f):
	weights_affine = net.state_dict()
	weights_affine = [weights_affine[k].double() for k in weights_affine]
	weights_affine = [get_homogeneous_weight(weights_affine[2 * i], weights_affine[2 * i + 1]) for i in
	                  range(len(weights_affine) // 2)]

	# eps after normalization?
	inputs = normalize(inputs.flatten().double())
	# homogeneous coordinates (append 1)
	l = torch.unsqueeze(torch.cat([inputs - eps, torch.ones(1, dtype=torch.float64)], 0), 1)
	u = torch.unsqueeze(torch.cat([inputs + eps, torch.ones(1, dtype=torch.float64)], 0), 1)

	weights_l = []
	weights_u = []
	is_affine_layers = []
	add_weights(weights_l, weights_u, is_affine_layers, l, u)

	for i in range(len(weights_affine)):
		add_weights(weights_l, weights_u, is_affine_layers, weights_affine[i])
		if i < len(weights_affine) - 1:
			l, u = back_substitution(weights_l, weights_u, is_affine_layers)
			w_l, w_u = get_spu_weights(l, u, f)
			add_weights(weights_l, weights_u, is_affine_layers, w_l, w_u)

	w_out = torch.eye(11, dtype=torch.float64)
	w_out = torch.cat([w_out[:true_label], w_out[true_label + 1:-1]], 0)
	w_out[:, true_label] -= 1.
	add_weights(weights_l, weights_u, is_affine_layers, w_out)
	l, u = back_substitution(weights_l, weights_u, is_affine_layers)

	# testcase 12: potential bug or numerical (rounding) problem?
	margin = 1e-1
	result = torch.all(u < -margin)
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

		wo = torch.tensor([[1., -1., 0.]], dtype=torch.float64)

		w1 = get_homogeneous_weight(w1, b1)
		w2 = get_homogeneous_weight(w2, b2)

		l = torch.unsqueeze(
			torch.cat([torch.tensor([0., 0.], dtype=torch.float64), torch.ones(1, dtype=torch.float64)], 0), 1)
		u = torch.unsqueeze(
			torch.cat([torch.tensor([1., 1.], dtype=torch.float64), torch.ones(1, dtype=torch.float64)], 0), 1)

		weights_l = []
		weights_u = []
		is_affine_layers = []
		add_weights(weights_l, weights_u, is_affine_layers, l, u)

		add_weights(weights_l, weights_u, is_affine_layers, w1)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)
		w_l, w_u = get_spu_weights(l, u, compute_linear_bounds_test)
		add_weights(weights_l, weights_u, is_affine_layers, w_l, w_u)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)
		add_weights(weights_l, weights_u, is_affine_layers, w2)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)
		add_weights(weights_l, weights_u, is_affine_layers, wo)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)

		result = torch.all(l >= 0.)
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

		wo = torch.tensor([[1., -1., 0]], dtype=torch.float64)

		w1 = get_homogeneous_weight(w1, b1)
		w2 = get_homogeneous_weight(w2, b2)
		w3 = get_homogeneous_weight(w3, b3)

		l = torch.unsqueeze(
			torch.cat([torch.tensor([-1., -1.], dtype=torch.float64), torch.ones(1, dtype=torch.float64)], 0), 1)
		u = torch.unsqueeze(
			torch.cat([torch.tensor([1., 1.], dtype=torch.float64), torch.ones(1, dtype=torch.float64)], 0), 1)

		weights_l = []
		weights_u = []
		is_affine_layers = []
		add_weights(weights_l, weights_u, is_affine_layers, l, u)

		add_weights(weights_l, weights_u, is_affine_layers, w1)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)
		w_l, w_u = get_spu_weights(l, u, compute_linear_bounds_test1)
		add_weights(weights_l, weights_u, is_affine_layers, w_l, w_u)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)
		add_weights(weights_l, weights_u, is_affine_layers, w2)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)
		w_l, w_u = get_spu_weights(l, u, compute_linear_bounds_test2)
		add_weights(weights_l, weights_u, is_affine_layers, w_l, w_u)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)
		add_weights(weights_l, weights_u, is_affine_layers, w3)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)
		add_weights(weights_l, weights_u, is_affine_layers, wo)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)

		result = torch.all(l > 0.)
		return result


	assert test_hw6()
	assert test_lec6()
