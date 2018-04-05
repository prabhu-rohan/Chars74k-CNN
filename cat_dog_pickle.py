from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


num_classes = 62  
image_size = 128  
pixel_depth = 255.0

def get_folders(path):
	data_folders = [os.path.join(path, d) for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
	#if len(data_folders) != num_classes:
    #    raise Exception(
    #      'Expected %d folders, one per class. Found %d instead.' % (
    #        num_classes, len(data_folders)))
	#	print(data_folders)
	return data_folders

train_folders=get_folders("C:\\Users\Rohan Prabhu\Desktop\Project 2\English\Fnt")

def load_letter(folder, min_num_images):
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
	print(folder)
	for image_index, image in enumerate(image_files):
		image_file = os.path.join(folder, image)
		try:
			image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[image_index, :, :] = image_data
		except IOError as e:
			print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
	
	num_images = image_index + 1
	dataset = dataset[0:num_images, :, :]
	if num_images < min_num_images:
		raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

	print('Full dataset tensor:', dataset.shape)
	print('Mean:', np.mean(dataset))
	print('Standard deviation:', np.std(dataset))
	return dataset
  
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
	dataset_names = []
	for folder in data_folders:
		set_filename = folder + '.pickle'
		dataset_names.append(set_filename)
		#if os.path.exists(set_filename) and not force:
		  # You may override by setting force=True.
		  #print('%s already present - Skipping pickling.' % set_filename)
		#else:
			#print('Pickling %s.' % set_filename)
			#try:
				#with open(set_filename, 'wb') as f:
				#	dataset = load_letter(folder, min_num_images_per_class)
				#	pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			##except Exception as e:
				#print('Unable to save data to', set_filename, ':', e)
		
	return dataset_names
  
train_datasets = maybe_pickle(train_folders, 1016)


def make_arrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels
	
def merge_datasets(pickle_files, train_size, valid_size,test_size):
	num_classes = len(pickle_files)
	valid_dataset, valid_labels = make_arrays(valid_size, image_size)
	train_dataset, train_labels = make_arrays(train_size, image_size)
	test_dataset, test_labels = make_arrays(test_size, image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes
	test_size_per_class=test_size//num_classes
	
	start_v, start_t,start_test = 0, 0 , 0
	end_v, end_t ,end_test= vsize_per_class, tsize_per_class,test_size_per_class
	end_l = vsize_per_class+tsize_per_class
	endf=vsize_per_class+tsize_per_class+test_size_per_class
	print('%d %d %d' %(vsize_per_class,tsize_per_class,test_size_per_class))
	for label, pickle_file in enumerate(pickle_files):
		try:
			with open(pickle_file, 'rb') as f:
				letter_set = pickle.load(f)
				# let's shuffle the letters to have random validation and training set
			np.random.shuffle(letter_set)
			if valid_dataset is not None:
				valid_letter = letter_set[:vsize_per_class, :, :]
				valid_dataset[start_v:end_v, :, :] = valid_letter
				valid_labels[start_v:end_v] = label
				start_v += vsize_per_class
				end_v += vsize_per_class

			train_letter = letter_set[vsize_per_class:end_l, :, :]
			train_dataset[start_t:end_t, :, :] = train_letter
			train_labels[start_t:end_t] = label
			start_t += tsize_per_class
			end_t += tsize_per_class
			
			test_letter = letter_set[end_l:endf, :, :]
			#print('%d %d' %(end_l,endf))
			test_dataset[start_test:end_test, :, :] = test_letter
			test_labels[start_test:end_test] = label
			start_test += test_size_per_class
			end_test += test_size_per_class
			#print('%d %d' %(start_test,end_test))
		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise
	return valid_dataset, valid_labels, train_dataset, train_labels,test_dataset,test_labels
  
train_size = 44020
valid_size = 9486
test_size = 9486
valid_dataset, valid_labels, train_dataset, train_labels, test_dataset,test_labels = merge_datasets(
  train_datasets, train_size, valid_size,test_size)
  
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

print(test_dataset)
print(test_labels)

np.save('train_dataset.npy',train_dataset)
np.save('train_labels.npy',train_labels)
np.save('test_dataset.npy',test_dataset)
np.save('test_labels.npy',test_labels)
np.save('valid_dataset.npy',valid_dataset)
np.save('valid_labels.npy',valid_labels)

#pickle_file = 'english.pickle'

#try:
 # f = open(pickle_file, 'wb')
  # 'train_dataset': train_dataset,
   # 'train_labels': train_labels,
    #'valid_dataset': valid_dataset,
   # 'valid_labels': valid_labels,
    #'test_dataset': test_dataset,
    #'test_labels': test_labels,
    #}
  #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  #f.close()
#except Exception as e:
#	print('Unable to save data to', pickle_file, ':', e)
#	raise
	


#statinfo = os.stat(pickle_file)
#print('Compressed pickle size:', statinfo.st_size)
