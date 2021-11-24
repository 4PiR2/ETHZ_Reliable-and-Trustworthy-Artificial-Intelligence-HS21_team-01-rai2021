import os
import sys

from verifier import main

argv_backup = sys.argv

print('testcase 12')
k = '1'
net = 'fc3'
spec = 'example_img1_0.00230.txt'
sys.argv = argv_backup + ['--net', 'net' + k + '_' + net, '--spec', '../test_cases/net' + k + '_' + net + '/' + spec]
main()

gt = {}
with open('../test_cases/gt.txt', 'r') as f:
	for line in f:
		fields = line.strip().split(',')
		gt[fields[0] + ',' + fields[1]] = fields[2]

for net in ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']:
	for k in ['0', '1']:
		for spec in os.listdir('../test_cases/net' + k + '_' + net):
			name = 'net' + k + '_' + net + ',' + spec
			print(name)
			print(gt[name], '(gt)')
			sys.argv = argv_backup + ['--net', 'net' + k + '_' + net, '--spec',
			                          '../test_cases/net' + k + '_' + net + '/' + spec]
			main()
