import numpy as np

class Conv2D:
	
	def __init__(self, inChannels, outChannels,  kernelSize, stride=1, padding=0):
		self.k = kernelSize
		self.s = stride
		self.p = padding

		self.weights = np.random.randn(outChannels, inChannels, self.k, self.k) * np.sqrt(2. / (inChannels * self.k * self.k))
		self.biases  = np.zeros(outChannels)
		self.cachedInput = None

	
	def forward(self, x):
		
		self.cachedInput = x
		batchSize, inC, inH, inW = x.shape
		outC, _, _, _ = self.weights.shape

		outH = (inH + 2 * self.p - self.k) // self.s + 1
		outW = (inW + 2 * self.p - self.k) // self.s + 1

		out = np.zeros((batchSize, outC, outH, outW))
		xPadded = np.pad(x, ((0,0), (0,0), (self.p, self.p), (self.p , self.p)), mode='constant')


		for b in range(batchSize):
			for c in range(outC):
				for i in range(outH):
					for j in range(outW):
						
						hStart = i * self.s
						hEnd = hStart + self.k
						wStart = j * self.s
						wEnd = wStart + self.k

						patch = xPadded[b, :, hStart:hEnd, wStart:wEnd]
						out[b, c, i, j] = np.sum(patch * self.weights[c]) + self.biases[c]

		return out
	

	def backward(self, dOut, lr):
		
		x = self.cachedInput
		batchSize, inC, inH, inW = x.shape
		outC, _, _, _ = self.weights.shape

		outH = dOut.shape[2]
		outW = dOut.shape[3]

		dWeights = np.zeros_like(self.weights)
		dBiases = np.zeros_like(self.biases)
		dX = np.zeros_like(x)
		xPadded = np.pad(x, ((0,0), (0,0), (self.p, self.p), (self.p , self.p)), mode='constant')
		dXPadded = np.zeros_like(xPadded)

		for b in range(batchSize):
			for c in range(outC):
				for i in range(outH):
					for j in range(outW):
						
						hStart = i * self.s
						hEnd = hStart + self.k
						wStart = j * self.s
						wEnd = wStart + self.k

						patch = xPadded[b, :, hStart:hEnd, wStart:wEnd]
						dWeights[c] += patch * dOut[b, c, i, j]
						dBiases[c]  += dOut[b, c, i, j]
						dXPadded[b, :, hStart:hEnd, wStart:wEnd] += self.weights[c] * dOut[b, c, i , j]


		if(self.p > 0):
			dX = dXPadded[:,  :, self.p:-self.p, self.p:-self.p]
		else:
			dX = dXPadded


		self.weights -= lr * dWeights
		self.biases  -= lr * dBiases
	



class ReLU:
		
	def __init__(self):
		self.cachedInput = None
	
	def forward(self, x):
		self.cachedInput = x
		return np.maximum(0, x)

	def backward(self, dOut, lr=None):
	
		dX = dOut.copy()
		dX[self.cachedInput <= 0] = 0
		return dX

	
class MaxPool2D:
	
	def __init__(self, poolSize=2, stride=2):
		
		self.p = poolSize
		self.s = stride
		self.cachedInput = None
		self.cachedArgmax = None


	def forward(self, x):
		
		self.cachedInput = x
		batchSize, channels, inH, inW = x.shape
		outH = (inH - self.p) // self.s + 1
		outW = (inW - self.p) // self.s + 1

		out = np.zeros((batchSize, channels, outH, outW))
		self.cachedArgmax = np.zeros((batchSize, channels, outH, outW, 2), dtype=int)


		for b in range(batchSize):
			for c in range(channels):
				for i in range(outH):
					for j in range(outW):
						
						hStart = i * self.s
						hEnd = hStart + self.p
						wStart = j * self.s
						wEnd = wStart + self.p

						patch = x[b, c, hStart:hEnd, wStart:wEnd]
						out[b, c, i, j] = np.max(patch)

						idx = np.unravel_index(np.argmax(patch), patch.shape)
						self.cachedArgmax[b, c, i, j] = [hStart + idx[0], wStart + idx[1]]


		return out
	

	def backward(self, dOut, lr=None):
		
		dX = np.zeros_like(self.cachedInput)
		batchSize, channels, outH, outW = dOut.shape

		for b in range(batchSize):
			for c in range(channels):
				for i in range(outH):
					for j in range(outW):
						
						maxH, maxW = self.cachedArgmax[b, c, i, j]
						dX[b, c, maxH, maxW] += dOut[b, c, i, j]

		return dX
	

class Dense:
	
	def __init__(self, inFeatures, outFeatures):
		self.weights = np.random.randn(inFeatures, outFeatures) * np.sqrt(2. / inFeatures)
		self.biases = np.zeros(outFeatures)
		self.cachedInput = None
		self.cachedShape = None


	def forward(self, x):
		
		self.cachedShape = x.shape
		if(len(x.shape) > 2):
			x = x.reshape(x.shape[0], -1)

		self.cachedInput = x
		return np.dot(x, self.weights) + self.biases
	
	def backward(self, dOut, lr):
		
		x = self.cachedInput
		dWeights = np.dot(x.T, dOut)
		dBiases = np.sum(dOut, axis=0)
		dX = np.dot(dOut, self.weights.T)

		self.weights -= lr * dWeights
		self.biases  -= lr * dBiases
	
		if(len(self.cachedShape) > 2):
			dX = dX.reshape(self.cachedShape)

		return dX
	


class CrossEntropyLoss:
	
	def __init__(self):
		self.cachedProbs  = None
		self.cachedLabels = None

	def forward(self, logits, labels):
		
		shiftedLogits = logits - np.max(logits, axis=1, keepdims=True)
		expScores = np.exp(shiftedLogits)
		probs = expScores / np.sum(expScores, axis=1, keepdims=True)

		self.cachedProbs  = probs
		self.cachedLabels = labels
		
		batchSize = logits.shape[0]

		correctLogProbs = -np.log(probs[np.arange(batchSize), labels] + 1e-7)
		dataLoss = np.sum(correctLogProbs) / batchSize
		
		return dataLoss


	def backward(self):

		batchSize = self.cachedProbs.shape[0]
		dLogits = self.cachedProbs.copy()

		dLogits[np.arange(batchSize), self.cachedLabels] -= 1
		dLogits /= batchSize
		
		return dLogits
	

class SequentialNetwork:

	def __init__(self, layers):
		self.layers = layers
	
	def forward(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		
		return x
	
	def backward(self, dOut, lr):
		for layer in reversed(self.layers):
			dOut = layer.backward(dOut, lr)

		return dOut
