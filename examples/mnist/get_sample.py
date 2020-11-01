import gzip
import os
import numpy as np
import struct

def get_sample_in(filename):
	with gzip.open(filename , 'rb') as file:
		buf = file.read()
	
	sample_in = []
	index = 0

	form = '>IIII'
	magic, numImages, numRows, numColumns = struct.unpack_from(form, buf, index)
	index += struct.calcsize(form)

	print("get_sample_in", filename)
	for i in range(numImages):
		form = '>784B'
		im_data = struct.unpack_from(form, buf, index)
		index += struct.calcsize(form)

		arr = np.array(im_data).astype("float32")
		arr = (arr - 128.0) / 128.0 
		sample_in.append(arr)
		
		if i % 10000 == 0: print("", i)

	return sample_in

def get_sample_out(filename):
	
	with gzip.open(filename , 'rb') as file:
		buf = file.read()
    
	sample_out = []
	index = 0
	form = '>II'
	magic, numItems = struct.unpack_from(form, buf, index)
	index += struct.calcsize(form)

	print("get_sample_out", filename)
	for i in range(numItems):
		form = '>1B'
		label_data = struct.unpack_from(form, buf, index)
		index += struct.calcsize(form)

		arr = np.array(label_data).astype("float32")
		sample_out.append(arr)

	return sample_out

train_in = get_sample_in('files/train-images-idx3-ubyte.gz')
train_out = get_sample_out("files/train-labels-idx1-ubyte.gz")

test_in = get_sample_in('files/t10k-images-idx3-ubyte.gz')
test_out = get_sample_out("files/t10k-labels-idx1-ubyte.gz")
