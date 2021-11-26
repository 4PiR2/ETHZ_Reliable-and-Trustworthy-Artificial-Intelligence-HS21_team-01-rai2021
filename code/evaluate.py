import os
import sys

from verifier import main

argv_backup = sys.argv

k = '0'
net = 'fc1'
spec = 'example_img1_0.05000.txt'
sys.argv = argv_backup + ['--net', 'net' + k + '_' + net, '--spec', '../test_cases/net' + k + '_' + net + '/' + spec]
result = main()

gts = {}
with open('../test_cases/gt.txt', 'r') as f:
	for line in f:
		fields = line.strip().split(',')
		gts[fields[0] + ',' + fields[1]] = fields[2]

score = 0
score_t = 0
for net in ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']:
	for k in ['0', '1']:
		for spec in os.listdir('../test_cases/net' + k + '_' + net):
			sys.argv = argv_backup + ['--net', 'net' + k + '_' + net, '--spec',
			                          '../test_cases/net' + k + '_' + net + '/' + spec]
			name = 'net' + k + '_' + net + ',' + spec
			gt = gts[name]
			result = 'verified' if main() else 'not verified'
			if result != gt:
				print(name)
				print(gt, '(gt)', '--', result, '(out)')
			if result == 'verified':
				if gt == 'verified':
					score += 1
				else:
					score -= 2
			if gt == 'verified':
				score_t += 1
print(score, '/', score_t)
