import torch
from torch import nn

from bs import get_homogeneous_weight, add_weights, back_substitution
from lb2 import lb_slope, is_intersect


class Net(nn.Module):
	def __init__(self, weights_affine, l, u, true_label, margin=0.):
		super().__init__()
		weights_affine = [get_homogeneous_weight(weights_affine[2 * i], weights_affine[2 * i + 1]) for i in
		                  range(len(weights_affine) // 2)]
		self.weights_affine = weights_affine
		l = torch.cat([l, torch.ones((1, 1), dtype=torch.float64)], 0)
		u = torch.cat([u, torch.ones((1, 1), dtype=torch.float64)], 0)
		self.inputs = [l, u]
		self.true_label = true_label
		self.margin = margin
		w_out = torch.eye(len(weights_affine[-1]), dtype=torch.float64)
		w_out = torch.cat([w_out[:true_label], w_out[true_label + 1:-1]], 0)
		w_out[:, true_label] -= 1.
		self.w_out = w_out
		self.spu_l_params = nn.ParameterList(
			[nn.Parameter(torch.zeros(len(w) - 1, dtype=torch.float64)) for w in weights_affine[:-1]])
		self.spu_u_params = nn.ParameterList(
			[nn.Parameter(torch.zeros(len(w) - 1, dtype=torch.float64)) for w in weights_affine[:-1]])
		self.verified_mask = torch.zeros(len(w_out), dtype=torch.bool)

	def get_spu_weights(self, l, u, spu_idx, f_init=None, f_clip=None):
		l, u = l.flatten()[:-1], u.flatten()[:-1]
		with torch.no_grad():
			if f_init is not None:
				(w_l, _), (w_u, _) = f_init(l, u)
				self.spu_l_params[spu_idx].copy_(w_l)
				self.spu_u_params[spu_idx].copy_(w_u)
			if f_clip is not None:
				w_l, w_u = self.spu_l_params[spu_idx], self.spu_u_params[spu_idx]
				b_l, b_u = lb_slope(l, u, w_l, w_u)
				(w_l_c, b_l_c), (w_u_c, b_u_c) = f_clip(l, u)
				self.spu_l_params[spu_idx].copy_(torch.where(is_intersect(l, u, w_l, b_l, w_l_c, b_l_c), w_l, w_l_c))
				self.spu_u_params[spu_idx].copy_(torch.where(is_intersect(l, u, w_u, b_u, w_u_c, b_u_c), w_u, w_u_c))
		w_l, w_u = self.spu_l_params[spu_idx], self.spu_u_params[spu_idx]
		b_l, b_u = lb_slope(l, u, w_l, w_u)
		W_l = get_homogeneous_weight(torch.diag(w_l), b_l)
		W_u = get_homogeneous_weight(torch.diag(w_u), b_u)
		return W_l, W_u

	def forward(self, f_init=None, f_clip=None):
		weights_l = []
		weights_u = []
		is_affine_layers = []
		add_weights(weights_l, weights_u, is_affine_layers, *self.inputs)

		for i in range(len(self.weights_affine)):
			add_weights(weights_l, weights_u, is_affine_layers, self.weights_affine[i])
			if i < len(self.weights_affine) - 1:
				l, u = back_substitution(weights_l, weights_u, is_affine_layers)
				if f_init is not None:
					w_l, w_u = self.get_spu_weights(l, u, i, f_init=f_init[i] if type(f_init) is list else f_init)
				elif f_clip is not None:
					w_l, w_u = self.get_spu_weights(l, u, i, f_clip=f_clip[i] if type(f_clip) is list else f_clip)
				else:
					w_l, w_u = self.get_spu_weights(l, u, i)
				add_weights(weights_l, weights_u, is_affine_layers, w_l, w_u)

		add_weights(weights_l, weights_u, is_affine_layers, self.w_out)
		l, u = back_substitution(weights_l, weights_u, is_affine_layers)
		self.verified_mask = torch.bitwise_or(u.flatten() < -self.margin, self.verified_mask)
		return u[~self.verified_mask]
