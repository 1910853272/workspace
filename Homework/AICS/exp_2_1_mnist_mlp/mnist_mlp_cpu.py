import os
import struct

import numpy as np

MNIST_DIR = "../mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"

def load_mnist(self,file_dir,is_images='True'):
    bin_file = open(file_dir,"rb")
    bin_data = bin_file.read()
    bin_file.close()
    if is_images:
        fmt_header = '>iiii'
        magic,num_images,num_rows, num_cols = struct.unpack(fmt_header,bin_data)
    else:
        fmt_header = '>ii'
        magic,num_images = struct.unpack(fmt_header,bin_data)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack('>' + str(data_size) + 'B',bin_data)
    mat_data = np.reshape(mat_data,(num_images,num_rows,num_cols))
    return mat_data

def load_data(self,file_dir,is_images='True'):
    print('Loading MNIST data from files...')
    train_images = self.load_mnist(os.path.join(MNIST_DIR,TRAIN_DATA),True)
    train_labels = self.load_mnist(os.path.join(MNIST_DIR,TRAIN_LABEL),False)
    test_images = self.load_mnist(os.path.join(MNIST_DIR,TEST_DATA),True)
    test_labels = self.load_mnist(os.path.join(MNIST_DIR,TEST_LABEL),False)
    self.train_data = np.append(train_images,train_labels,axis=1)
    self.test_data = np.append(test_images,test_labels,axis=1)
    

