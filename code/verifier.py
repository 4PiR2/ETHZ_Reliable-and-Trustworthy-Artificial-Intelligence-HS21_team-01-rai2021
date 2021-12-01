import argparse
import torch
from networks import FullyConnected

import time
import torch.optim as optim

from lb import lb_random_mix, lb_base, lb_boxlike, lb_parallelogram
from bs import analyze_f
from bs2 import Net

DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, eps, true_label):
	weights_affine = net.state_dict()
	weights_affine = [weights_affine[k].double() for k in weights_affine]
	l = torch.maximum(inputs - eps, torch.zeros(inputs.shape))
	u = torch.minimum(inputs + eps, torch.ones(inputs.shape))
	l = net.layers[:2](l).T.double()
	u = net.layers[:2](u).T.double()

	net = Net(weights_affine, l, u, true_label, margin=1e-11)
	optimizer = optim.Adam(net.parameters(), lr=1e-1)
	_ = net.forward(f_init=lb_boxlike)
	result = net.forward(f_init=lb_base)
	ts = time.time()
	count = 0
	while len(result) > 0:
		optimizer.zero_grad()
		loss = result[0]
		loss.backward()
		optimizer.step()
		result = net.forward()
		count += 1
		if time.time() - ts > 61:
			break
	if len(result) == 0:
		# print(count, 'v')
		return True
	# print(count, 'nv')
	return False


def analyze_old(net, inputs, eps, true_label):
	weights_affine = net.state_dict()
	weights_affine = [weights_affine[k].double() for k in weights_affine]
	n_spu_layers = len(weights_affine) // 2 - 1
	l = torch.maximum(inputs - eps, torch.zeros(inputs.shape))
	u = torch.minimum(inputs + eps, torch.ones(inputs.shape))
	l = net.layers[:2](l).T.double()
	u = net.layers[:2](u).T.double()
	result_t = torch.zeros(10 - 1, dtype=torch.bool)
	f_list = [lb_base, lb_boxlike, lb_parallelogram]
	f_list += [[*([lb_base] * (n_spu_layers - 1)), lambda l, u: lb_random_mix(l, u, [lb_base, lb_boxlike])],
	           [*([lb_boxlike] * (n_spu_layers - 1)), lambda l, u: lb_random_mix(l, u, [lb_base, lb_boxlike])]] * 1000
	# for i in range(10 + 1):
	# 	for j in range(10 + 1):
	# 		f_list += [lambda l, u: compute_linear_bounds(l, u, i / 10, j / 10)]
	for f in f_list:
		result = analyze_f(weights_affine, l, u, true_label, f)
		result_t = torch.bitwise_or(result, result_t)
		if torch.all(result_t):
			return True
	return False


def main():
	parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
	parser.add_argument('--net',
	                    type=str,
	                    required=True,
	                    help='Neural network architecture which is supposed to be verified.')
	parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
	args = parser.parse_args()

	with open(args.spec, 'r') as f:
		lines = [line[:-1] for line in f.readlines()]
		true_label = int(lines[0])
		pixel_values = [float(line) for line in lines[1:]]
		eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

	if args.net.endswith('fc1'):
		net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
	elif args.net.endswith('fc2'):
		net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
	elif args.net.endswith('fc3'):
		net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
	elif args.net.endswith('fc4'):
		net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
	elif args.net.endswith('fc5'):
		net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
	else:
		assert False

	net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

	inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
	outs = net(inputs)
	pred_label = outs.max(dim=1)[1].item()
	assert pred_label == true_label

	result = analyze(net, inputs, eps, true_label)
	if __name__ != '__main__':
		return result
	if result:
		print('verified')
	else:
		print('not verified')


if __name__ == '__main__':
	main()
