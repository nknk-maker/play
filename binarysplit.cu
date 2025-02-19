#pragma once
#include <cuda_runtime.h>
#include <BigInt.cu>

using namespace std;

__global__ void ksplit(BigInt* x, BigInt* y, BigInt* z, int siz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = 1<<siz;
    int hl = l>>1;
    int pos = i*l;
    if (pos + hl >= n) return;
    BigInt nx = x[pos] * x[pos+hl];
    BigInt ny = x[pos+hl] * y[pos] + y[pos+hl] * z[pos];
    BigInt nz = z[pos] * z[pos+hl];
}

void binary_split(BigInt* x, BigInt *y, BigInt *z, int n) {
    for (int i = 1; (1<<i) <= n; i++) {
        int blockSize = 32;
        int numBlocks = (n>>i + blockSize - 1) / blockSize;
        ksplit<<<numBlocks, blockSize>>>(x, y, z, i, n);
    }
}