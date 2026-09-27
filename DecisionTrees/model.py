import ctypes as ct
import numpy as np
from numpy.ctypeslib import ndpointer as ndp
import os


libPath = os.path.join(os.path.dirname(__file__), 'libtree.so')
lib = ct.CDLL(libPath)

lib.fit.argtypes = [
	ndp(ct.c_float, flags="C_CONTIGUOUS"),
	ndp(ct.c_int32, flags="C_CONTIGUOUS"),
	ct.c_uint32, ct.c_uint32, ct.c_uint32,
	ct.c_uint32, ct.c_uint32
]

lib.fit.restype = ct.c_void_p

lib.predict.argtypes = [
    ct.c_void_p,
    ndp(ct.c_float, flags="C_CONTIGUOUS"),
    ndp(ct.c_int32, flags="C_CONTIGUOUS"),
    ct.c_uint32, ct.c_uint32
]

lib.freeDecisionTree.argtypes = [ct.c_void_p]


class DecisionTree:
	
	def __init__(self, maxDepth=5, minSamplesSplit=2):
		
		self.maxDepth = maxDepth
		self.minSamplesSplit = minSamplesSplit
		self.treePtr = None
	

	def fit(self, X, y):
		
		print("[model.py] : Inside fit(X, y)")
		Xc = np.ascontiguousarray(X, dtype=np.float32)
		yc = np.ascontiguousarray(y, dtype=np.int32)
		

		numSamples, numFeatures = Xc.shape
		numClasses = len(np.unique(yc))
		
		print("[model.py] : Passing to C")
		print(f"numSamples     : {numSamples}")
		print(f"numFeatures    : {numFeatures}")
		print(f"numClasses     : {numClasses}")
		print(f"maxDepth       : {self.maxDepth}")
		print(f"minSampleSplit : {self.minSamplesSplit}")
		self.treePtr = lib.fit(
			Xc, yc, 
			ct.c_uint32(numSamples), 
			ct.c_uint32(numFeatures),
			ct.c_uint32(numClasses),
			ct.c_uint32(self.maxDepth), 
			ct.c_uint32(self.minSamplesSplit)
		)
		print("[model.py] : Back from C into fit")
		print("[model.py] : End of fit")


	def predict(self, X):
		
		if not self.treePtr:
			raise ValueError("Tree not fitted")


		Xc = np.ascontiguousarray(X, dtype=np.float32)
		numSamples, numFeatures = Xc.shape

		yPred = np.zeros(numSamples, dtype=np.int32)

		lib.predict(
			self.treePtr, Xc, yPred, 
			ct.c_uint32(numSamples), 
			ct.c_uint32(numFeatures)
		)

		return yPred
	

	def __del__(self):
		if self.treePtr:
			lib.freeDecisionTree(self.treePtr)
