import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os

# 1. Load the compiled C library natively into Python
lib_path = os.path.join(os.path.dirname(__file__), 'libengine.so')
lib = ctypes.CDLL(lib_path)

# 2. Map the exact Argument Types for the C functions
lib.initEngine.argtypes = [
    ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # layer_sizes
    ctypes.c_uint32, # num_layers
    ctypes.c_float   # l2lambda
]

lib.train.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # x_raw
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # y_raw
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, # rows, x_cols, y_cols
    ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32   # epochs, lr, batch_size
]

lib.cleanupEngine.argtypes = []

lib.saveWeights.argtypes = [ctypes.c_char_p]
lib.loadWeights.argtypes = [ctypes.c_char_p]

lib.predict.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # x_raw
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # y_pred_raw
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32  # rows, x_cols, y_cols
]

# 3. Create the PyTorch-Style Wrapper Class
class NeuralNetwork:

    def __init__(self, layer_sizes, l2lambda=0.0):
        # Force sizes to flat, contiguous 32-bit unsigned integers
        self.sizes = np.array(layer_sizes, dtype=np.uint32)
        # Boot up the C Engine Arenas
        lib.initEngine(self.sizes, len(self.sizes), l2lambda)

    def train(self, X, y, epochs=100, lr=0.001, batch_size=64):
        # Enforce 32-bit floats and C-contiguous memory layout
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        y_c = np.ascontiguousarray(y, dtype=np.float32)
        
        # Calculate matrix dimensions
        rows, x_cols = X_c.shape
        _, y_cols = y_c.shape
        
        # Fire the C Engine! Data stays in Python RAM, C reads it in place.
        lib.train(
            X_c, y_c, 
            rows, x_cols, y_cols, 
            epochs, lr, batch_size
        )

    def save(self, filepath):
        # Convert Python string to C-compatible UTF-8 bytes
        c_filepath = filepath.encode('utf-8')
        lib.saveWeights(c_filepath)

    def load(self, filepath):
        # Convert Python string to C-compatible UTF-8 bytes
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weight file {filepath} does not exist.")
        c_filepath = filepath.encode('utf-8')
        lib.loadWeights(c_filepath)

    def predict(self, X):
        # 1. Enforce 32-bit floats and contiguous memory layout
        X_c = np.ascontiguousarray(X, dtype=np.float32)
        rows, x_cols = X_c.shape

        # 2. Determine the output size from the architecture sizes (last layer)
        y_cols = self.sizes[-1]

        # 3. CALLER ALLOCATES: Python creates the empty result array
        y_pred = np.zeros((rows, y_cols), dtype=np.float32)

        # 4. Fire the C Engine! It fills y_pred in-place using Scratch Arena memory.
        lib.predict(X_c, y_pred, rows, x_cols, y_cols)

        return y_pred

    def __del__(self):
        # When this Python object is garbage collected, wipe the C memory!
        lib.cleanupEngine()
