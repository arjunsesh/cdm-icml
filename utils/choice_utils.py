# This file contains some utils to help load and play around with choice data
import pdb
import csv
import re
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
import ast

def choice_str_to_int(generic_input):
	"""
	Changes the choice from a string "I would take ticket 1" to an int (1)

	Update: didn't see that slot_chosen was a column, don't really need this.
	Inputs: generic_input - can be one of:
	1) a str (single instance to process), returns an integer with the choice value. 
	2) a list, returns a new list with integers
	3) a dict, modifies inplace the list under 'choice' label and returns None
	Outputs:
	Specified above given the option
	"""
	if type(generic_input) is str:
		item = generic_input
		return int(item([-1]))
	elif type(generic_input) is list:
		choice_list = generic_input 
		return [int(item[-1]) for item in choice_list]
	elif type(generic_input) is dict:
		# method simply modifies dict and returns none
		data_dict = generic_input
		data_dict['choice'] = [int(item[-1]) for item in data_dict['choice']]
		return None
	else:
		raise TypeError('Type of input must be a string, list, or dict')

def handle_days(generic_input):
	"""
	Changes the d*_text from "In * days" to the integer *. 

	Update: didn't see
	that the d's were their own column, don't really need this.
	Inputs: generic_input - can be one of:
	1) a str (single instance to process), returns an integer with number of days. 
	2) a list, returns a new list with integers containing the days
	3) a dict, modifies inplace the list under 'choice' label and returns None
	Outputs:
	Specified above given the option
	"""
	if type(generic_input) is str:
		item = generic_input
		match = re.match(r'In (.*) days', item)
		if match:
			return int(match.group(1))
		else:
			return 0
	elif type(generic_input) is list:
		d_list = generic_input
		return [int(re.match(r'In (.*) days', line).group(1))
			if re.match(r'In (.*) days', line) else 0 for line in d_list]
	elif type(generic_input) is dict:
		# method does not return anything here, only modifies dict
		data_dict = generic_input
		for n in range(1,4):
			key = 'd'+str(n)+'_text'
			data_dict[key] = [int(re.match(r'In (.*) days', line).group(1))
			if re.match(r'In (.*) days', line) else 0 for line in data_dict[key]]
		return None
	else:
		raise TypeError('Type of input must be a string, list, or dict')

def feature_map(input):
	"""
	Maps tuples of variables into a unique index

	Features are sometimes more than one variable, but for the purpose of
	embeddings, it is best to index them by one variable - this maps collections of
	variables to an index. Ignores None values
	Inputs: input - a list of lists where each list is a feature over all data.
	The code forms tuples from each lists, and adds a mapping to an index, and 
	returns that mapping as a dict. The lists must be of same size

	Outputs: dictionary containing mapping.

	"""
	data_map = {};
	map_idx = 0
	data_len = len(input[0])
	for data_list in input:
		assert data_len == len(data_list)

	for data_idx in range(data_len):
		datapoint = ()
		for idx in range(len(input)):
			datapoint += (input[idx][data_idx],)
		
		if datapoint not in data_map and not any(
				map(lambda x: x is None, datapoint)):
			data_map[datapoint] = map_idx
			map_idx += 1

	return data_map

def dataset_headers(dataset_keyword, dense=False):
	"""
	Produces headers for feature types in each dataset, given the dataset.

	Is just a mapping from the dataset keyword, to its corresponding feature headers
	Inputs: dataset_keyword (ex. 'risk', 'time', 'coop')
	dense (optional): for 'coop', provides additional (redundant) features that
	correspond to the differences in reward

	Outputs: the list of feature headers

	"""
	if dataset_keyword == 'risk':
		return ['x', 'p']
	elif dataset_keyword == 'time':
		return ['x', 'd']
	elif dataset_keyword == 'coop':
		if dense:
			return ['you', 'other', 'adv_ineq', 'dis_adv_ineq']
		else:
			return ['you', 'other']
	elif 'assu' in dataset_keyword:
		return ['item']
	elif 'snack' in dataset_keyword:
		return ['item']
	elif 'ctxt_jd' in dataset_keyword:
		return ['item']
	elif 'syn_nc' in dataset_keyword:
		return ['item']
	elif 'syn_c' in dataset_keyword:
		return ['cov0', 'cov1']
	elif 'syn' in dataset_keyword: # intentionally comes after 'syn_c'
		return ['item']
	elif 'SF' in dataset_keyword:
		return ['mode']
	else:
		raise ValueError('Invalid keyword for dataset')

def handle_chosen_slot(data_dict):
	"""
	Decides the correct key for the given data_dict for the chosen_slot entry

	While the column is named 'chosen_slot' in the risk data set, it is named
	'slot_chosen' in the time and coop datasets. This function produces the correct
	key for the scenario.
	Inputs: data_dict - the dictionary containing the dataset

	Outputs:
	key - a str, either 'chosen_slot' or 'slot_chosen'

	"""
	key = 'chosen_slot'
	try:
		data_dict[key]
	except:
		key = 'slot_chosen'
	return key
def cluster_features(data_map, k=50):
	"""
	This function should only (generally) be called by feature id when the 
	purpose is to analyze the choice set distribution of a featured choice
	dataset (like risk, time, coop, etc). For those datasets, often there are
	lots of unique datapoints for the given data size, so its better to place
	small balls in the feature space and id those datapoints based on those
	balls. We will use k-means to do this efficiently.
	"""
	import sklearn.cluster # importing here because this library is not usually
						# installed in machines, and because fcn is rarely used.
	feature_list = np.array([list(key) for key in data_map.keys()])
	kmeans=sklearn.cluster.KMeans(n_clusters=k,n_init=100,max_iter=3000).fit(feature_list)
	for idx,key in enumerate(data_map.keys()):
		data_map[key] = kmeans.labels_[idx]
	return data_map
def feature_id(data_dict, features, context_size=None, cluster=False, k=50):
	"""
	Adds adds feature_ids to the data in data_dict.

	This maps collections of variables in a list to an index, making it useful for
	embeddings. Maps None values to the largest index, incremented (num_features)
	Inputs: data_dict - dict containing data, with features labled by headers
	features - list of headers eg. ['x','p'] that contain feature types
	context_size (optional) - Can limit max size. 
	Size of the choice context, so it sweeps for all the features in the data

	Outputs: data_map, the mapping between ids and features
	         (also modifies data_dict in place)
	"""
	# Figure out max choice set length:
	if context_size is None:
		matches = [re.match(features[0]+r'(\d+)', k) for k in data_dict.keys()]
		context_size = max([int(m.group(1)) for m in matches if m])
	temp_list = []
	ismultiset = False
	for i in range(len(features)):
		temp_list.append([])
		for j in range(1,context_size+1):
			temp_list[-1] += data_dict[features[i]+str(j)]
	data_map = feature_map(temp_list)
	if cluster:
		data_map = cluster_features(data_map,k)
	num_features = len(data_map.values())

	data_dict['context_ids'] = []
	data_dict['choice_id'] = []
	data_dict['context_ids_wo_choice'] = []
	data_dict['choice_set_lengths'] = []
	for j in range(len(data_dict[features[0]+'1'])):
		data_dict['context_ids'].append([])
		data_dict['context_ids_wo_choice'].append([])
		counter = 0
		for i in range(1,context_size+1):
			datapoint = ()
			for k in range(len(features)):
				datapoint += (data_dict[features[k]+str(i)][j],)
			if any(map(lambda x: x is None, datapoint)):
				ismultiset=True
				data_dict['context_ids'][-1].append(num_features)
				data_dict['context_ids_wo_choice'][-1].append(num_features)
			else:
				counter += 1
				data_dict['context_ids'][-1].append(data_map[datapoint])
				data_dict['context_ids_wo_choice'][-1].append(
															data_map[datapoint])
		chosen_slot_key = handle_chosen_slot(data_dict)
		data_dict['choice_id'].append(data_dict['context_ids'][-1][data_dict[chosen_slot_key][j]])
		del data_dict['context_ids_wo_choice'][-1][data_dict[chosen_slot_key][j]]
		data_dict['choice_set_lengths'].append(counter)
	
	return data_map, context_size, ismultiset


def read_data(filename, pre_proc=True):
	"""
	Reads csv data files and returns a dict of lists.

	The main method used to process csv file data, takes in the csv file and
	converts it into a data dictionary, with headers corresponding to the headers
	in the csv file and the columns stored as lists. Replaces empty strings with
	None values.
	Inputs:
	filename - string file name of data
	pre_proc (optional) - preprocesses the data to convert some string data into
	integer valued data for access.

	Outputs: data_dict that contains file data

	"""
	output = {}
	with open(filename, mode='r') as infile:
		reader = csv.DictReader(infile)
		for row in reader:
			for k, v in row.items():
				if v != '':
					try:
						output.setdefault(k,[]).append(int(v))
					except:
						output.setdefault(k,[]).append(v)
				else:
					output.setdefault(k,[]).append(None)

	if pre_proc:
		for key in output:
			if key == 'choice':
				output[key] = choice_str_to_int(output[key])
			elif re.match(r'd._text', key):
				output[key] = handle_days(output[key])
			elif key == '':
				output['index'] = output.pop(key)


	return output

def read_and_id_data(dataset, filename=None, pre_proc=True, context_size=None,
					cluster=False, k=50):
	# Wrapper for read_data and feature_id that makes this process easier
	if filename is None:
		filename = '../data/raw/'+dataset+'_data_final.csv'
	data_dict = read_data(filename,pre_proc)
	dh = dataset_headers(dataset)
	data_map, context_size, ismultiset = feature_id(data_dict,dh,context_size,
													cluster,k)

	return data_dict, data_map, context_size, ismultiset
def segment_data(data_dict, m, train_size, val_size, seed=None, n_fold_cv=1):
	"""
	Segments data dictionary into train, validation, and test dictionaries.

	Goes key by key to split the dataset, which is input as a dictionary (after
	running feature_id from choice_utils on it)

	Inputs:
	data_dict - dataset, specified as dictionary
	m - int, length of data dictionary (total number of points, from which the 
	number of test points is computed)
	train_size - int, number of training points
	val_size - int, number of validation points
	seed - controls the random seed
	n_fold_cv - decides whether to return multiple folds
	Outputs:
	data_dict_tr_list - list of training data_dict
	data_dict_val_list - list of validation data_dict
	data_dict_te_list - list of test data_dict

	"""
	cross_val = (n_fold_cv > 1)
	data_dict_tr_list,data_dict_val_list, data_dict_te_list = [],[],[]
	if cross_val:
		assert m == train_size + val_size, "there is no test set in n_fold_cv"

	if seed is not None:
		current_random_state = np.random.get_state()
		np.random.seed(seed) # set seed according to specification
		idx_order = np.random.choice(m, train_size + val_size, False)
		np.random.set_state(current_random_state) # restore to where it was
												  # pre-seed
	else:
		idx_order = np.random.choice(m, train_size + val_size, False)
	for i in range(n_fold_cv):
		data_dict_tr, data_dict_te,data_dict_val = {},{},{}
		idx_val = idx_order[i*val_size:(i+1)*val_size]

		idx_train = np.concatenate([idx_order[:i*val_size],
													idx_order[(i+1)*val_size:]])
		for key in data_dict.keys():
			data_dict_tr[key] = np.array(data_dict[key])[idx_train].tolist()
			data_dict_val[key] = np.array(data_dict[key])[idx_val].tolist()
			data_dict_te[key] = data_dict_val[key]
			if not cross_val: # everything not tr or val is test
				idx_test =  np.setdiff1d(np.arange(m),idx_order,True)
				data_dict_te[key] = np.array(data_dict[key])[idx_test].tolist()
		data_dict_tr_list.append(data_dict_tr)
		data_dict_val_list.append(data_dict_val)
		data_dict_te_list.append(data_dict_te)

	return data_dict_tr_list, data_dict_val_list, data_dict_te_list


def choice_set_map(data_dict, data_map, dataset_headers):
	choice_set_map = {}
	for idx, _ in enumerate(data_dict[dataset_headers[0]+'1']):
		choice_set = tuple(data_map[tuple(data_dict[header+str(j)][idx] 
						for header in dataset_headers)] for j in range(1,4))
		try:
			choice_set_map[choice_set] += 1
		except:
			choice_set_map[choice_set] = 1

	return choice_set_map


def get_current_file_suffix(file_name, file_type, file_dir='./'):
	
	file_list = os.listdir(file_dir)
	name_to_match = file_name + r'(.*)' + file_type
	file_suffixes = [re.match(name_to_match, item).group(1)
						for item in file_list if re.match(name_to_match, item)]
	if len(file_suffixes) == 0:
		return ''
	elif len(file_suffixes) == 1:
		return '_1'
	else:
		file_suffixes.remove('')
		file_ids = [int(suffix[1]) for suffix in file_suffixes]
		return '_' + str(max(file_ids) + 1)

def split_samples_helper(data_dict, num_features):
	samples = []
	for idx, c_set in enumerate(data_dict['context_ids']):
		choice_set = tuple(sorted([c for c in c_set if c is not None and c != num_features]))
		sort_chosen_slot = choice_set.index(data_dict['choice_id'][idx])
		samples.append((choice_set, sort_chosen_slot))
	return samples
def split_samples(dataset, nep, split=.25, alpha=.1):
	# Code used to merge CDM data load process with PCMC loading process
	data_dict = read_data('../data/raw/'+dataset+'_data_final.csv')
	headers = dataset_headers(dataset)
	data_map, mcsl, ismultiset = feature_id(data_dict, headers)
	num_features = len(data_map.values())
	m = len(data_dict['index'])
	data_dict_tr, data_dict_val, data_dict_te = segment_data(data_dict, m, 
										int(.8*m), int(.1*m), 2)
	
	trainsamples = split_samples_helper(data_dict_tr, num_features)
	testsamples = split_samples_helper(data_dict_te, num_features)
	#Rest is just code from the orignal split_samples
	Ctest = {}
	for (S,choice) in testsamples:
		if S not in Ctest:
			Ctest[S]=np.ones(len(S))*0.#alpha
		Ctest[S][choice]+=1

	trainlist = [{} for i in range(nep)]
	a = len(trainsamples)/nep
	for i in range(nep):
		for (S,choice) in trainsamples[i*a:(i+1)*a]:
			if S not in trainlist[i]:
				trainlist[i][S]=np.ones(len(S))*alpha
			trainlist[i][S][choice]+=1
	
	return trainlist,Ctest