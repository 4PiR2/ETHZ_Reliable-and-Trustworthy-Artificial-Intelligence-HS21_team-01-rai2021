import argparse
import torch
from networks import FullyConnected

# import time
import torch.optim as optim

from lb import lb_base, lb_boxlike
from bs import Net

DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, eps, true_label):
	weights_affine = net.state_dict()
	weights_affine = [weights_affine[k].double() for k in weights_affine]
	l = torch.maximum(inputs - eps, torch.zeros(inputs.shape))
	u = torch.minimum(inputs + eps, torch.ones(inputs.shape))
	l = net.layers[:2](l).T.double()
	u = net.layers[:2](u).T.double()

	net_deep_poly = Net(weights_affine, l, u, true_label)
	optimizer = optim.Adam(net_deep_poly.parameters(), lr=1e-1)
	verified_mask = torch.zeros(9, dtype=torch.bool)
	margin = 1e-11

	# ts = time.time()

	for f in [lb_base, lb_boxlike]:
		result = net_deep_poly.forward(f_init=f)
		verified_mask = torch.bitwise_or(result < -margin, verified_mask)

	for i in range(9):
		# if verified_mask[i]:
		# 	continue
		# result = net_deep_poly.forward(f_init=lb_boxlike)
		# verified_mask = torch.bitwise_or(result < -margin, verified_mask)
		while not verified_mask[i]:
			optimizer.zero_grad()
			result[i].backward()
			optimizer.step()
			result = net_deep_poly.forward()
			verified_mask = torch.bitwise_or(result < -margin, verified_mask)
			# if time.time() - ts > 61:
			# 	return False
	return True


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
