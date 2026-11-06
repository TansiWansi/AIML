from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from cnn import *

print("Loading 8x8 digits dataset...")
digits = load_digits()

# Reshape to (batch, channels, height, width)
X = digits.images.reshape((len(digits.images), 1, 8, 8)) 
X = X / 16.0 # Normalize pixels to [0, 1]
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Graph
model = SequentialNetwork([
    # Input: (1, 8, 8) -> Output: (8, 8, 8)
    Conv2D(inChannels=1, outChannels=8, kernelSize=3, stride=1, padding=1),
    ReLU(),
    # Input: (8, 8, 8) -> Output: (8, 4, 4)
    MaxPool2D(poolSize=2, stride=2),
    # Input: (8, 4, 4) -> Flattened to 128 -> Output: 10 classes
    Dense(inFeatures=8 * 4 * 4, outFeatures=10)
])

loss_fn = CrossEntropyLoss()
learning_rate = 0.05
epochs = 10
batch_size = 32

print("Initiating Training Loop...")

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # 1. Forward Pass
        logits = model.forward(X_batch)
        
        # 2. Compute Loss
        loss = loss_fn.forward(logits, y_batch)
        total_loss += loss * len(X_batch)
        
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == y_batch)
        
        # 3. Backward Pass (Backprop)
        d_logits = loss_fn.backward()
        model.backward(d_logits, learning_rate)
        
    train_acc = correct / len(X_train)
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(X_train):.4f} | Accuracy: {train_acc*100:.2f}%")
