import numpy as np
import time

def im2col(x, fieldHeight, fieldWidth, padding=1, stride=1):
    cChannel, height, width = x.shape
    outHeight = (height + 2 * padding - fieldHeight) // stride + 1
    outWidth = (width + 2 * padding - fieldWidth) // stride + 1
    
    xPadded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    cols = np.zeros((cChannel * fieldHeight * fieldWidth, outHeight * outWidth), dtype=x.dtype)
    
    colIdx = 0
    for c in range(cChannel):
        for kh in range(fieldHeight):
            for kw in range(fieldWidth):
                # Vectorized 2D strided slice across all spatial output locations
                slice2d = xPadded[c, kh : kh + outHeight * stride : stride, kw : kw + outWidth * stride : stride]
                cols[colIdx, :] = slice2d.ravel()
                colIdx += 1
                
    return cols

def conv2dGemm(x, w, b, padding=1, stride=1):
    cOut, cIn, kH, kW = w.shape
    _, height, width = x.shape
    
    outH = (height + 2 * padding - kH) // stride + 1
    outW = (width + 2 * padding - kW) // stride + 1

    xCol = im2col(x, kH, kW, padding, stride)
    wRow = w.reshape(cOut, -1)
    
    out = np.dot(wRow, xCol) + b.reshape(-1, 1)
    return out.reshape(cOut, outH, outW)

def conv2dNaive(x, w, b, padding=1, stride=1):
    cOut, cIn, kH, kW = w.shape
    _, height, width = x.shape
    
    outH = (height + 2 * padding - kH) // stride + 1
    outW = (width + 2 * padding - kW) // stride + 1
    
    out = np.zeros((cOut, outH, outW))
    xPadded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    for c in range(cOut):
        for i in range(outH):
            for j in range(outW):
                hStart = i * stride
                hEnd = hStart + kH
                wStart = j * stride
                wEnd = wStart + kW
                
                patch = xPadded[:, hStart:hEnd, wStart:wEnd]
                out[c, i, j] = np.sum(patch * w[c]) + b[c]
                
    return out

if __name__ == "__main__":
    np.random.seed(42)
    
    x = np.random.randn(3, 32, 32)
    w = np.random.randn(16, 3, 3, 3)
    b = np.random.randn(16)
    
    print("Executing Naive Convolution...")
    start = time.time()
    outNaive = conv2dNaive(x, w, b, padding=1, stride=1)
    tNaive = time.time() - start
    
    print("Executing im2col GEMM Convolution...")
    start = time.time()
    outGemm = conv2dGemm(x, w, b, padding=1, stride=1)
    tGemm = time.time() - start
    
    isEqual = np.allclose(outNaive, outGemm, atol=1e-5)
    
    print(f"\nParity Check: {'PASSED' if isEqual else 'FAILED'}")
    print(f"Naive Time: {tNaive:.5f}s")
    print(f"GEMM Time:  {tGemm:.5f}s")
    print(f"Speedup:    {tNaive / tGemm:.2f}x faster")
