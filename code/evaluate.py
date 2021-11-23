import os
import sys

from verifier import main

argv_backup = sys.argv
for net in ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']:
	os.environ['net'] = net
	for k in ['0', '1']:
		os.environ['k'] = k
		print('Evaluating network net' + k + '_' + net)
		for spec in os.listdir('../test_cases/net' + k + '_' + net):
			os.environ['spec'] = spec
			sys.argv = argv_backup + ['--net', 'net' + k + '_' + net, '--spec',
			                          '../test_cases/net' + k + '_' + net + '/' + spec]
			main()
