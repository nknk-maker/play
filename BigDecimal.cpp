#pragma once

constexpr int NUMBER = 1<<15;
constexpr int blockSize = 1024;
constexpr int numBlocks = (NUMBER+ blockSize - 1) / blockSize;

#ifdef __CUDA_ARCH__

#include <vector>
#include <iostream>
#include <assert.h>
#include <convolve.cpp>
#include <cuda_runtime.h>
using namespace std;

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

__global__ void kset(int* a, int p, int q, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i+p >= n) return;
    a[i+p] = q;
}

struct BigDecimal {
    int siz = NUMBER,*u;
    BigDecimal() {
        cudaMalloc(&u, sizeof(int)*siz);
        cudaMemset(u, 0, sizeof(int)*siz);
    }
    BigDecimal(const int& b) {
        cudaMalloc(&u, sizeof(int)*siz);
        cudaMemset(u, 0, sizeof(int)*siz);
        int p = 0, x = b;
        vector<int> v(siz);
        while (x > 0) {
            set(p+siz/2, x%10);
            x /= 10;
            p++;
        }
    }
    BigDecimal(const long long& b) {
        cudaMalloc(&u, sizeof(int)*siz);
        cudaMemset(u, 0, sizeof(int)*siz);
        int p = 0; long long x = b;
        vector<int> v(siz);
        while (x > 0) {
            set(p+siz/2, x%10);
            x /= 10;
            p++;
        }
    }
    BigDecimal(const BigDecimal& b) : siz(b.siz) {
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
    const BigDecimal operator=(const BigDecimal& b) {
        cudaMemcpy(u, b.u, sizeof(int)*siz, cudaMemcpyDeviceToDevice);
        return *this;
    }
    const BigDecimal operator=(const int& b) {
        int p = 0, x = b;
        cudaMemset(u, 0, sizeof(int)*siz);
        vector<int> v(siz);
        while (x > 0) {
            set(p+siz/2, x%10);
            x /= 10;
            p++;
        }
        return *this;
    }
    const BigDecimal operator=(const long long& b) {
        int p = 0; long long x = b;
        cudaMemset(u, 0, sizeof(int)*siz);
        vector<int> v(siz);
        while (x > 0) {
            set(p+siz/2, x%10);
            x /= 10;
            p++;
        }
        return *this;
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
        return *this * rhs;
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
        BigDecimal x, _2;
        // 初期値を決める
        int n = siz-1;
        vector<int> v(siz);
        cudaMemcpy(v.data(), b.u, sizeof(int)*siz, cudaMemcpyDeviceToHost);
        while (v[n] == 0) n--;
        n = siz - n - 1;

        x.set(n, 1); // 0.1
        for (int k = 1; k < siz/2; k *= 2) {
            x = x * (_2 - b * x);
        }
        *this = (*this) * x;
        val();
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const BigDecimal& b) {
        vector<int> v(b.siz);
        cudaMemcpy(v.data(), b.u, sizeof(int)*b.siz, cudaMemcpyDeviceToHost);
        while (v[i] == 0) i--;
        for (;i >= b.siz/2; i--) os << v[i];
        os << '.';
        for (;i >= 0; i--) os << v[i];
        return os;
    }
};

#else

#include <vector>
#include <iostream>
#include <assert.h>
#include <convolve.cpp>
using namespace std;

struct BigDecimal {
    int siz = NUMBER;
    vector<int> v;
    BigDecimal() {
        v.assign(siz, 0);
    }
    BigDecimal(const int& b) {
        v.assign(siz, 0);
        int p = 0, x = b;
        while (x > 0) {
            v[p+siz/2] = x%10;
            x /= 10;
            p++;
        }
    }
    BigDecimal(const long long& b) {
        v.assign(siz, 0);
        int p = 0; long long x = b;
        while (x > 0) {
            v[p+siz/2] = x%10;
            x /= 10;
            p++;
        }
    }
    BigDecimal(const BigDecimal& b) : siz(b.siz), v(b.v) {}
    const BigDecimal val() {
        for (int i = 0; i < siz-1; i++) {
            int p = v[i] / 10;
            if (v[i] < 0) p--;
            v[i+1] += p;
            v[i] -= p * 10;
        }
        return *this;
    }
    void set(int i, int a) {
        v[i] = a;
    }
    const BigDecimal operator=(const BigDecimal& b) {
        v = b.v;
        return *this;
    }
    const BigDecimal operator=(const int& b) {
        int p = 0, x = b;
        v.assign(siz, 0);
        while (x > 0) {
            v[p+siz/2] = x%10;
            x /= 10;
            p++;
        }
        return *this;
    }
    const BigDecimal operator=(const long long& b) {
        int p = 0; long long x = b;
        v.assign(siz, 0);
        while (x > 0) {
            v[p+siz/2] = x%10;
            x /= 10;
            p++;
        }
        return *this;
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
        return *this * rhs;
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
        for (int i = 0; i < siz; i++) v[i] += b.v[i];
        val();
        return (*this);
    }
    const BigDecimal &operator-=(const BigDecimal& b) {
        for (int i = 0; i < siz; i++) v[i] -= b.v[i];
        val();    
        return (*this);
    }
    const BigDecimal &operator*=(const BigDecimal& b) {
        assert(siz == b.siz);
        v = convolve(v, b.v);
        val();
        return *this;
    }
    const BigDecimal &operator/=(BigDecimal b) {
        assert(siz == b.siz);
        BigDecimal x;
        BigDecimal _2;
        _2.v[siz/2] = 2;
        // 初期値を決める
        int n = siz-1;
        while (b.v[n] == 0) n--;
        n = siz - n - 1;
        x.v[n] = 1;
        
        for (int k = 1; k <= siz; k *= 2) {
            x = x * (_2 - b * x);
        }
        *this = (*this) * x;
        val();
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const BigDecimal& b) {
        int i = b.siz-1;
        while (b.v[i] == 0) i--;
        for (;i >= b.siz/2; i--) os << b.v[i];
        os << '.';
        for (;i >= 0; i--) os << b.v[i];
        return os;
    }
};

#endif