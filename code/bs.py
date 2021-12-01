import torch


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
	l, u = weights_l[-1], weights_u[-1]
	n_nodes = len(l)
	for i in range(len(weights_l) - 1)[::-1]:
		if is_affine_layers is not None and is_affine_layers[i]:
			l @= weights_l[i]
			u @= weights_l[i]
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


def analyze_f(weights_affine, l, u, true_label, f, margin=0.):
	weights_affine = [get_homogeneous_weight(weights_affine[2 * i], weights_affine[2 * i + 1]) for i in
	                  range(len(weights_affine) // 2)]
	# homogeneous coordinates (append 1)
	l = torch.cat([l, torch.ones((1, 1), dtype=torch.float64)], 0)
	u = torch.cat([u, torch.ones((1, 1), dtype=torch.float64)], 0)

	weights_l = []
	weights_u = []
	is_affine_layers = []
	add_weights(weights_l, weights_u, is_affine_layers, l, u)

	for i in range(len(weights_affine)):
		# l, u = back_substitution(weights_l, weights_u, is_affine_layers)  # debug
		add_weights(weights_l, weights_u, is_affine_layers, weights_affine[i])
		if i < len(weights_affine) - 1:
			l, u = back_substitution(weights_l, weights_u, is_affine_layers)
			w_l, w_u = get_spu_weights(l, u, f[i] if type(f) is list else f)
			add_weights(weights_l, weights_u, is_affine_layers, w_l, w_u)

	# l, u = back_substitution(weights_l, weights_u, is_affine_layers)  # debug
	w_out = torch.eye(len(weights_affine[-1]), dtype=torch.float64)
	w_out = torch.cat([w_out[:true_label], w_out[true_label + 1:-1]], 0)
	w_out[:, true_label] -= 1.
	add_weights(weights_l, weights_u, is_affine_layers, w_out)
	l, u = back_substitution(weights_l, weights_u, is_affine_layers)
	return (u < -margin).flatten()


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

		result = analyze_f([w1, b1, w2, b2], torch.tensor([[0.], [0.]], dtype=torch.float64),
		                   torch.tensor([[1.], [1.]], dtype=torch.float64), 0, compute_linear_bounds_test, -1e10)
		return torch.all(result)


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

		result = analyze_f([w1, b1, w2, b2, w3, b3], torch.tensor([[-1.], [-1.]], dtype=torch.float64),
		                   torch.tensor([[1.], [1.]], dtype=torch.float64), 0,
		                   [compute_linear_bounds_test1, compute_linear_bounds_test2])
		return torch.all(result)


	assert test_hw6()
	assert test_lec6()
