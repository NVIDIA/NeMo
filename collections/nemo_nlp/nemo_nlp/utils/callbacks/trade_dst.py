# Copyright (c) 2019 NVIDIA Corporation
__all__ = ['eval_iter_callback', 'eval_epochs_done_callback']

import os
import random
import time

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from nemo.utils.exp_logging import get_logger

logger = get_logger('')


def eval_iter_callback(tensors,
					   global_vars,
					   eval_data_layer):
	# print(tensors)
	# print(global_vars)
	if 'loss' not in global_vars:
		global_vars['loss'] = []
	if 'point_outputs' not in global_vars:
		global_vars['point_outputs'] = []
	if 'gate_outputs' not in global_vars:
		global_vars['gate_outputs'] = []
	if 'gating_labels' not in global_vars:
		global_vars['gating_labels'] = []
	if 'gate_outputs' not in global_vars:
		global_vars['gate_outputs'] = []

	for kv, v in tensors.items():
		if kv.startswith('loss'):
			global_vars['loss'].append(v[0].cpu().numpy())
		if kv.startswith('point_outputs'):
			global_vars['point_outputs'].append(v[0].cpu().numpy())
		if kv.startswith('gate_outputs'):
			global_vars['gate_outputs'].append(v[0].cpu().numpy())


def list2str(l):
	return ' '.join([str(j) for j in l])


def eval_epochs_done_callback(global_vars, graph_fold):
	print(global_vars['loss'])
	return {}


def evaluate(self, dev, matric_best, slot_temp, early_stop=None):
	# Set to not-training mode to disable dropout
	self.encoder.train(False)
	self.decoder.train(False)
	print("STARTING EVALUATION")
	all_prediction = {}
	inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
	pbar = tqdm(enumerate(dev), total=len(dev))
	for j, data_dev in pbar:
		# Encode and Decode
		batch_size = len(data_dev['context_len'])
		_, gates, words, class_words = self.encode_and_decode(data_dev, False, slot_temp)

		for bi in range(batch_size):
			if data_dev["ID"][bi] not in all_prediction.keys():
				all_prediction[data_dev["ID"][bi]] = {}
			all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {"turn_belief": data_dev["turn_belief"][bi]}
			predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
			gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

			# pointer-generator results
			if args["use_gate"]:
				for si, sg in enumerate(gate):
					if sg == self.gating_dict["none"]:
						continue
					elif sg == self.gating_dict["ptr"]:
						pred = np.transpose(words[si])[bi]
						st = []
						for e in pred:
							if e == 'EOS':
								break
							else:
								st.append(e)
						st = " ".join(st)
						if st == "none":
							continue
						else:
							predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))
					else:
						predict_belief_bsz_ptr.append(slot_temp[si] + "-" + inverse_unpoint_slot[sg.item()])
			else:
				for si, _ in enumerate(gate):
					pred = np.transpose(words[si])[bi]
					st = []
					for e in pred:
						if e == 'EOS':
							break
						else:
							st.append(e)
					st = " ".join(st)
					if st == "none":
						continue
					else:
						predict_belief_bsz_ptr.append(slot_temp[si] + "-" + str(st))

			all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

			if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
				print("True", set(data_dev["turn_belief"][bi]))
				print("Pred", set(predict_belief_bsz_ptr), "\n")

	if args["genSample"]:
		json.dump(all_prediction, open("all_prediction_{}.json".format(self.name), 'w'), indent=4)

	joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr", slot_temp)

	evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr, "Joint F1": F1_score_ptr}
	print(evaluation_metrics)

	# Set back to training mode
	self.encoder.train(True)
	self.decoder.train(True)

	joint_acc_score = joint_acc_score_ptr  # (joint_acc_score_ptr + joint_acc_score_class)/2
	F1_score = F1_score_ptr

	if (early_stop == 'F1'):
		if (F1_score >= matric_best):
			self.save_model('ENTF1-{:.4f}'.format(F1_score))
			print("MODEL SAVED")
		return F1_score
	else:
		if (joint_acc_score >= matric_best):
			self.save_model('ACC-{:.4f}'.format(joint_acc_score))
			print("MODEL SAVED")
		return joint_acc_score


def evaluate_metrics(all_prediction, from_which, slot_temp):
	total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
	for d, v in all_prediction.items():
		for t in range(len(v)):
			cv = v[t]
			if set(cv["turn_belief"]) == set(cv[from_which]):
				joint_acc += 1
			total += 1

			# Compute prediction slot accuracy
			temp_acc = compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
			turn_acc += temp_acc

			# Compute prediction joint F1 score
			temp_f1, temp_r, temp_p, count = compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
			F1_pred += temp_f1
			F1_count += count

	joint_acc_score = joint_acc / float(total) if total != 0 else 0
	turn_acc_score = turn_acc / float(total) if total != 0 else 0
	F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
	return joint_acc_score, F1_score, turn_acc_score


def compute_acc(gold, pred, slot_temp):
	miss_gold = 0
	miss_slot = []
	for g in gold:
		if g not in pred:
			miss_gold += 1
			miss_slot.append(g.rsplit("-", 1)[0])
	wrong_pred = 0
	for p in pred:
		if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
			wrong_pred += 1
	ACC_TOTAL = len(slot_temp)
	ACC = len(slot_temp) - miss_gold - wrong_pred
	ACC = ACC / float(ACC_TOTAL)
	return ACC


def compute_prf(gold, pred):
	TP, FP, FN = 0, 0, 0
	if len(gold) != 0:
		count = 1
		for g in gold:
			if g in pred:
				TP += 1
			else:
				FN += 1
		for p in pred:
			if p not in gold:
				FP += 1
		precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
		recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
		F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
	else:
		if len(pred) == 0:
			precision, recall, F1, count = 1, 1, 1, 1
		else:
			precision, recall, F1, count = 0, 0, 0, 1
	return F1, recall, precision, count
