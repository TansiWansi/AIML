import numpy as np

class Node:
	
	def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
		
		self.threshold = threshold
		self.feature = feature
		self.left = left
		self.right = right
		self.value = value


class DecisionTree:
	
	def __init__(self, maxDepth=10, minSamplesSplit=2):
		
		self.maxDepth = maxDepth
		self.minSamplesSplit = minSamplesSplit
		self.root = None
	

	def _gini(self, y):
		count = np.bincount(y)
		probs = count / len(y)

		return 1.0 - np.sum(probs ** 2)

	
	def _split(self, X, feature, threshold):
		leftIdx = np.argwhere(X[:, feature] <= threshold).flatten()
		rightIdx = np.argwhere(X[:, feature] > threshold).flatten()

		return leftIdx, rightIdx

		
	def _bestSplit(self, X, y):
		
		bestGini = float("inf")
		bestFeat, bestThresh = None, None
		numSamples, numFeatures = X.shape

		for feature in range(numFeatures):
			
			thresholds = np.unique(X[:, feature])
			for thresh in thresholds:
				
				leftIdx, rightIdx = self._split(X, feature, thresh)
				if(len(leftIdx) == 0 or len(rightIdx) == 0): continue

				giniLeft = self._gini(y[leftIdx])
				giniRight = self._gini(y[rightIdx])

				giniSplit = (len(leftIdx) / numSamples) * giniLeft + (len(rightIdx) / numSamples) * giniRight

				if giniSplit < bestGini:
					bestGini = giniSplit
					bestFeat = feature
					bestThresh = thresh



		return bestFeat, bestThresh
	

	def _buildTree(self, X, y, depth=0):
		
		numSamples, numLabels = len(y), len(np.unique(y))

		if(numLabels == 1 or numSamples < self.minSamplesSplit or depth >= self.maxDepth):
			leafValue = np.argmax(np.bincount(y))
			return Node(value=leafValue)

		
		feature, thresh = self._bestSplit(X, y)
		if(feature is None):
			leafValue = np.argmax(np.bincount(y))
			return Node(value=leafNode)


		leftIdx, rightIdx = self._split(X, feature, thresh)
		left = self._buildTree(X[leftIdx], y[leftIdx], depth + 1)
		right = self._buildTree(X[rightIdx], y[rightIdx], depth + 1)

		return Node(feature=feature, threshold=thresh, left=left, right=right)
	

	def fit(self, X, y):
		y = y.astype(np.int32)
		self.root = self._buildTree(X, y)
	

	def _traverse(self, x, node : Node):
		if(node.value is not None):
			return node.value

		if(x[node.feature] <= node.threshold):
			return self._traverse(x, node.left)

		return self._traverse(x, node.right)
	

	def predict(self, X):
		return np.array([self._traverse(x, self.root) for x in X])
