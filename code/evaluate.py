# 00 net0_fc1 example_img0_0.01800.txt verified
# 01 net0_fc1 example_img1_0.05000.txt verified
# 02 net1_fc1 example_img0_0.07500.txt verified
# 03 net1_fc1 example_img1_0.07200.txt verified
# 04 net0_fc2 example_img0_0.09500.txt verified
# 05 net0_fc2 example_img1_0.08300.txt verified
# 06 net1_fc2 example_img0_0.05200.txt not verified
# 07 net1_fc2 example_img1_0.07200.txt verified
# 08 net0_fc3 example_img0_0.07500.txt verified
# 09 net0_fc3 example_img1_0.08100.txt verified
# 10 net1_fc3 example_img0_0.06100.txt verified
# 11 net1_fc3 example_img1_0.00230.txt not verified
# 12 net0_fc4 example_img0_0.03300.txt verified
# 13 net0_fc4 example_img1_0.01800.txt not verified
# 14 net1_fc4 example_img0_0.05200.txt not verified
# 15 net1_fc4 example_img1_0.01300.txt verified
# 16 net0_fc5 example_img0_0.02100.txt not verified
# 17 net0_fc5 example_img1_0.01900.txt verified
# 18 net1_fc5 example_img0_0.08400.txt verified
# 19 net1_fc5 example_img1_0.07800.txt verified
# 20 net0_fc1 final_img4_0.05700.txt not verified
# 21 net0_fc1 final_img5_0.01400.txt verified
# 22 net1_fc1 final_img2_0.09100.txt verified
# 23 net1_fc1 final_img5_0.02950.txt not verified
# 24 net1_fc1 final_img9_0.05200.txt verified
# 25 net0_fc2 final_img0_0.07400.txt not verified
# 26 net0_fc2 final_img6_0.08200.txt verified
# 27 net1_fc2 final_img2_0.15100.txt verified
# 28 net1_fc2 final_img5_0.04400.txt not verified
# 29 net1_fc2 final_img9_0.15100.txt verified
# 30 net0_fc3 final_img0_0.15100.txt verified
# 31 net0_fc3 final_img3_0.15300.txt verified
# 32 net0_fc3 final_img9_0.12600.txt verified
# 33 net1_fc3 final_img2_0.07600.txt verified
# 34 net1_fc3 final_img8_0.06100.txt not verified
# 35 net0_fc4 final_img2_0.02750.txt verified
# 36 net0_fc4 final_img9_0.03500.txt verified
# 37 net1_fc4 final_img2_0.02500.txt not verified
# 38 net1_fc4 final_img5_0.02300.txt not verified
# 39 net1_fc4 final_img8_0.01300.txt verified
# 40 net0_fc5 final_img2_0.03500.txt verified
# 41 net0_fc5 final_img6_0.02900.txt verified
# 42 net0_fc5 final_img9_0.02850.txt verified
# 43 net1_fc5 final_img3_0.09800.txt verified
# 44 net1_fc5 final_img6_0.03500.txt verified

test_list_full = range(45)
test_list_10 = range(20)
test_list_prelim = range(20, 45)
test_list_v = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 15, 17, 18, 19, 21, 22, 24, 26, 27, 29, 30, 31, 32, 33, 35, 36, 39,
               40, 41, 42, 43, 44]
test_list_nv = [6, 11, 13, 14, 16, 20, 23, 25, 28, 34, 37, 38]

test_list = [27, 30, 31]
# test_list = test_list_v

import os
import sys
import time

from verifier import main

argv_backup = sys.argv

test_cases = []

with open('../test_cases/gt.txt', 'r') as f:
	for line in f:
		fields = line.strip().split(',')
		t = {'id': '{:0>2}'.format(len(test_cases)), 'dir': '../test_cases', 'net': fields[0], 'test case': fields[1],
		     'ground truth': fields[2]}
		test_cases.append(t)

with open('../prelim_results.txt', 'r') as f:
	is_title_line = True
	for line in f:
		if is_title_line:
			is_title_line = False
			continue
		fields = line.strip().split(',')
		if len(fields) != 6:
			continue
		t = {'id': '{:0>2}'.format(len(test_cases)), 'dir': '../prelim_test_cases', 'net': fields[0],
		     'test case': fields[1], 'ground truth': fields[2], 'your output': fields[3], 'your runtime': fields[4],
		     'your point': fields[5]}
		test_cases.append(t)

print('*** Results ***\n')
score = 0
score_total = 0
for i in test_list:
	t = test_cases[i]
	sys.argv = argv_backup + ['--net', t['net'], '--spec', os.path.join(t['dir'], t['net'], t['test case'])]
	if t['ground truth'] == 'not verified':
		continue
	print(t['id'], t['net'], t['test case'])
	ts = time.time()
	t['result'] = 'verified' if main() else 'not verified'
	print(t['ground truth'], '(gt)', '--', t['result'], '(out)', '--', time.time() - ts)

print('\n\n*** Summary ***\n')
for i in test_list:
	t = test_cases[i]
	if 'result' not in t:
		continue
	if t['result'] == 'verified':
		if t['ground truth'] == 'verified':
			score += 1
		else:
			score -= 2
	else:
		print(t['id'], t['net'], t['test case'])
		print(t['ground truth'], '(gt)', '--', t['result'], '(out)')
	if t['ground truth'] == 'verified':
		score_total += 1
print('Score:', score, '/', score_total)
