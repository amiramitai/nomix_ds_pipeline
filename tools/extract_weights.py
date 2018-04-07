# import the inspect_checkpoint library
# from tensorflow.python.tools import inspect_checkpoint as chkp

import sys
from tensorflow.python import pywrap_tensorflow
import numpy as np
import os

reader = pywrap_tensorflow.NewCheckpointReader(sys.argv[1])
var_to_shape_map = reader.get_variable_to_shape_map()
for key in sorted(var_to_shape_map):
    print("tensor_name: ", key)
    out_file = os.path.join(sys.argv[2], key.replace('/', '_'))
    np.save(out_file, reader.get_tensor(key))

    # print(reader.get_tensor(key))
# print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file(sys.argv[1], tensor_name='', all_tensors=True, all_tensor_names=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# print only tensor v2 in checkpoint file
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]