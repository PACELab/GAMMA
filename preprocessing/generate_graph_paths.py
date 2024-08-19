import os
import sys
import pickle


def read_paths_from_file(filename):
    with open(filename, 'r') as f:
        paths = f.readlines()
    return [path.strip() for path in paths]


def process_graph_paths():
	
	filepath = "/home/ad1238/bottleneck_detection_microservices/newExperiment/Data/realistic_july31_25min_repeat_800_5/processed_traces/graph_paths_1"

	paths = read_paths_from_file(filepath)

	source_all = []
	target_all = []

	for path in paths:
		nodes = path.split('->')
		
		source_all.extend(nodes[:-1])
		target_all.extend(nodes[1:])

	source = []
	target = []

	for sa, ta in zip(source_all, target_all):
		toggle_add = 1
		if len(source) > 0 and len(target) > 0:
			for s,t in zip(source, target):
				if  s == sa and t == ta:
					toggle_add = 0
					break       
		
		if toggle_add == 1:
			source.append(sa)
			target.append(ta)


	return source, target




if __name__ == "__main__":

	source, target = process_graph_paths()
	print(source, target)
	# print(len(source), len(target))

	with open('/home/ad1238/bottleneck_detection_microservices/newExperiment/util_data/source.pkl', 'wb') as f:
		pickle.dump(source, f)

	with open('/home/ad1238/bottleneck_detection_microservices/newExperiment/util_data/target.pkl', 'wb') as f:
		pickle.dump(target, f)