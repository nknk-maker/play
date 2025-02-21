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
    if (i+1 < n) a[i+1] += p;
    a[i] -= p * 10;
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

__global__ void kshift(int* a, int* b, int d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (0 <= i+d && i+d < n) b[i] = a[i+d];
    else b[i] = 0;
}

__global__ void kcp(int *a, int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = b[i];
}

struct BigInt {
    int siz, blockSize, numBlocks;
    int *u;
    BigInt(int n) : siz(n) {
        cudaMalloc(&u, sizeof(int)*siz);
        cudaMemset(u, 0, sizeof(int)*siz);
        blockSize = 1024;
        numBlocks = (siz + blockSize - 1) / blockSize;
    }
    const BigInt val() {
        kval<<<numBlocks, blockSize>>>(u, siz);
        return *this;
    }
    void set(int i, int a) {
        kset<<<1, 1>>>(u, i, a, siz);
    }
    const BigInt shift(int k) {
        int *y;
        cudaMalloc(&y, sizeof(int)*siz);
        kshift<<<numBlocks, blockSize>>>(u, y, k, siz);
        return *this;
    }
    int comp(const BigInt& b) {
        int *p, *q;
        cudaMalloc(&p, sizeof(int));
        cudaMalloc(&q, sizeof(int));
        *p = -1; *q = 0;
        kcomp<<<numBlocks, blockSize>>>(u, b.u, p, q);
        return *q;
    }
    BigInt operator=(const BigInt& b) {
        siz = b.siz;
        kcp<<<numBlocks, blockSize>>>(u, b.u, min(siz, b.siz));
        return *this;
    }
    bool operator<(const BigInt& b) {
        return comp(b) == -1;
    }
    bool operator>(const BigInt& b) {
        return comp(b) == 1;
    }
    bool operator==(const BigInt& b) {
        return comp(b) == 0;
    }
    bool operator!=(const BigInt& b) {
        return !((*this) == b);
    }
    bool operator<=(const BigInt& b) {
        return comp(b) != 1;
    }
    bool operator>=(const BigInt& b) {
        return comp(b) != -1;
    }
    const BigInt operator+(const BigInt& b) const { 
        BigInt ret = *this;
        return ret += b;
    }
    const BigInt operator-(const BigInt& b) const { 
        BigInt ret = *this;
        return ret -= b; 
    }
    const BigInt operator*(const BigInt& b) const { 
        BigInt ret = *this;
        return ret *= b; 
    }
    const BigInt operator/(const BigInt& b) const { 
        BigInt ret = *this;
        return ret /= b; 
    }
    const BigInt &operator+=(const BigInt& b) {
        kadd<<<numBlocks, blockSize>>>(u, b.u, min(siz, b.siz));
        val();
        return (*this);
    }
    const BigInt &operator-=(const BigInt& b) {
        ksub<<<numBlocks, blockSize>>>(u, b.u, min(siz, b.siz));
        val();    
        return (*this);
    }
    const BigInt &operator*=(BigInt b) {
        assert(siz == b.siz);
        u = convolve(u, b.u, siz);
        val();
        return *this;
    }
    const BigInt &operator/=(BigInt b) {
        assert(siz == b.siz);
        BigInt x(siz), _2(siz), a = *this;
        x.set(0, 1); // 0.1
        for (int k = 1; k < siz/2; k *= 2) {
            _2.set(k, 2);
            x = x * (_2 - b * x);
            _2.set(k, 0);
        }
        *this = (*this) * x;
        shift(siz/2);
        // if (a >= *this * b) b++;
        return *this;
    }
};

ostream& operator<<(ostream& os, const BigInt& b) {
    int pt = b.siz-1;
    vector<int> v(b.siz);
    cudaMemcpy(v.data(), b.u, sizeof(int)*b.siz, cudaMemcpyDeviceToHost);
    while (v[pt] == 0 && pt >= 1) pt--; pt++;
    while (pt--) os << v[pt];
    return os;
}

int main() {
    cin.tie(nullptr);
    ios_base::sync_with_stdio(false);
    BigInt a(NUMBER);
    BigInt b(NUMBER);
    // a.set(0, 9);
    a.set(1, 1);
    b.set(0, 2);
    // b.set(1, 9);
    a /= b;
    cout << a << endl;
}

