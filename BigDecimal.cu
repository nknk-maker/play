#pragma once
#include <vector>
#include <iostream>
#include <assert.h>
#include <convolve.cu>
#include <cuda_runtime.h>
using namespace std;
#define NUMBER 1<<3


__global__ void kval(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int p = a[i] / 10;
    if (a[i] < 0) p--;
    if (i+1 < n) atomicAdd(&a[i+1], p);
    atomicSub(&a[i], p * 10);
}

__global__ void kadd(int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

__global__ void ksub(int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] -= b[i];
}

__global__ void kcomp(int* a, int* b, int* p, int* q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (*p > i && *q != 0) return;
    if (a[i] > b[i]) *p = i, *q = 1;
    if (a[i] < b[i]) *p = i, *q = -1;
}

__global__ void kset(int* a, int p, int q, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i+p >= n) return;
    a[i+p] = q;
}

__global__ void kcp(int *a, int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = b[i];
}

struct BigDecimal {
    int siz, blockSize, numBlocks;
    int *u;
    BigDecimal(int n) : siz(n) {
        cudaMalloc(&u, sizeof(int)*siz);
        cudaMemset(u, 0, sizeof(int)*siz);
        blockSize = 1024;
        numBlocks = (siz + blockSize - 1) / blockSize;
    }
    BigDecimal(const BigDecimal& b) : siz(b.siz), blockSize(b.blockSize), numBlocks(b.numBlocks) {
        cudaMalloc(&u, sizeof(int)*siz);
        cudaMemcpy(u, b.u, sizeof(int)*siz, cudaMemcpyDeviceToDevice);
    }
    const BigDecimal val() {
        kval<<<numBlocks, blockSize>>>(u, siz);
        return *this;
    }
    void set(int i, int a) {
        kset<<<1, 1>>>(u, i, a, siz);
    }
    int comp(const BigDecimal& b) {
        int *p, *q;
        cudaMalloc(&p, sizeof(int));
        cudaMalloc(&q, sizeof(int));
        *p = -1; *q = 0;
        kcomp<<<numBlocks, blockSize>>>(u, b.u, p, q);
        return *q;
    }
    const BigDecimal operator=(const BigDecimal& b) {
        kcp<<<numBlocks, blockSize>>>(u, b.u, siz);
        return *this;
    }
    const BigDecimal operator=(const int& b) {
        int p = 0, x = b;
        cudaMemset(u, 0, sizeof(int)*siz);
        while (x > 0) {
            set(p+siz/2, x%10);
            x /= 10;
            p++;
        }
    }
    const BigDecimal operator+(const int& b) const {
        BigDecimal rhs = b;
        return *this + rhs;
    }
    const BigDecimal operator-(const int& b) const {
        BigDecimal rhs = b;
        return *this - rhs;
    }
    const BigDecimal operator*(const int& b) const {
        BigDecimal rhs = b;
        return *this * b;
    }
    const BigDecimal operator/(const int& b) const {
        BigDecimal rhs = b;
        return *this / rhs;
    }
    const BigDecimal operator+(const BigDecimal& b) const { 
        BigDecimal ret = *this;
        return ret += b;
    }
    const BigDecimal operator-(const BigDecimal& b) const { 
        BigDecimal ret = *this;
        return ret -= b; 
    }
    const BigDecimal operator*(const BigDecimal& b) const { 
        BigDecimal ret = (*this);
        return ret *= b; 
    }
    const BigDecimal operator/(const BigDecimal& b) const { 
        BigDecimal ret = *this;
        return ret /= b; 
    }
    const BigDecimal &operator+=(const BigDecimal& b) {
        kadd<<<numBlocks, blockSize>>>(u, b.u, min(siz, b.siz));
        val();
        return (*this);
    }
    const BigDecimal &operator-=(const BigDecimal& b) {
        ksub<<<numBlocks, blockSize>>>(u, b.u, min(siz, b.siz));
        val();    
        return (*this);
    }
    const BigDecimal &operator*=(const BigDecimal& b) {
        assert(siz == b.siz);
        int* r = convolve(u, b.u, siz);
        cudaMemcpy(u, r+siz/2, sizeof(int)*siz, cudaMemcpyDeviceToDevice);
        cudaFree(r);
        val();
        return *this;
    }
    const BigDecimal &operator/=(BigDecimal b) {
        assert(siz == b.siz);
        BigDecimal x(siz);
        // 初期値を決める
        int n = siz-1;
        int v[siz];
        cudaMemcpy(v, b.u, sizeof(int)*siz, cudaMemcpyDeviceToHost);
        while (v[n] == 0) n--;
        n = siz - n - 1;

        x.set(n, 1); // 0.1
        for (int k = 1; k < siz/2; k *= 2) {
            BigDecimal _2(siz);
            _2.set(siz/2, 2);
            x = x * (_2 - b * x);
        }
        *this = (*this) * x;
        val();
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const BigDecimal& b) {
        vector<int> v(b.siz);
        cudaMemcpy(v.data(), b.u, sizeof(int)*b.siz, cudaMemcpyDeviceToHost);
        for (int i = b.siz-1; i >= b.siz/2; i--) os << v[i];
        os << '.';
        for (int i = b.siz/2-1; i >= 0; i--) os << v[i];
        return os;
    }
};
