#pragma once

#ifdef __CUDA_ARCH__

#include <thrust/complex.h>
#include <cuda_runtime.h>

using namespace std;
using COMP = thrust::complex<double>;

__global__ void cfft(COMP* v, int n, int siz, bool inv) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int l = 1<<siz;
    int hl = l>>1;
    int g = id/hl;
    int pos = g*l + id%hl;
    if (pos+hl< n) {
        COMP a = v[pos], b = v[pos+hl];
        double angle = 2.0*M_PI*(id%hl)/l;
        if (!inv) angle *= -1;
        COMP phase = COMP(cos(angle), sin(angle));
        b *= phase;
        v[pos] = a+b;
        v[pos+hl] = a-b;
    }
}

__global__ void kbit_rev(COMP* v, int m, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    int j = 0, x = id;
    for (int i = 0; i < m; i++) j = (j << 1) | (x & 1), x >>= 1;
    if (j > id) {
        COMP tmp = v[id];
        v[id] = v[j];
        v[j] = tmp;
    }
}

__global__ void kmul(COMP* v, COMP *u, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    v[id] *= u[id];
}

__global__ void kdiv(COMP* v, int m, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    v[id] /= m;
}

__global__ void rtoi(COMP* v, int* u, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    u[id] = __double2int_rn(v[id].real());
}

__global__ void itor(int* v, COMP* u, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    u[id] = COMP(__int2double_rn(v[id]), 0.0);
}

void fft(COMP* v, int n, bool inv=false) {
    int m = 0; while ((1<<m)<n) m++;
    int blockSize = 32;
    int numBlocks = (n/2 + blockSize - 1) / blockSize;
    kbit_rev<<<numBlocks*2, blockSize>>>(v, m, n); // bit反転はn個のコアで行う必要がある
    for (int siz = 1; (1<<siz) <= n; siz++) cfft<<<numBlocks, blockSize>>>(v, n, siz, inv);
    return;
}

int* convolve(int* f_, int* g_, int siz) {
    int n = siz*2;
    int blockSize = 32;
    int numBlocks = (n + blockSize - 1) / blockSize;

    COMP *f, *g;
    cudaMalloc(&f, sizeof(COMP)*n);
    cudaMalloc(&g, sizeof(COMP)*n);
    // 配列全体をゼロ初期化
    cudaMemset(f, 0, sizeof(COMP) * n);
    cudaMemset(g, 0, sizeof(COMP) * n);
    itor<<<numBlocks, blockSize>>>(f_, f, siz);
    itor<<<numBlocks, blockSize>>>(g_, g, siz);

    fft(f, n);
    fft(g, n);

    kmul<<<numBlocks, blockSize>>>(f, g, n);
    fft(f, n, true);
    kdiv<<<numBlocks, blockSize>>>(f, n, n);

    int* r;
    cudaMalloc(&r, sizeof(int)*n);
    rtoi<<<numBlocks, blockSize>>>(f, r, n);
    cudaFree(f); cudaFree(g);

    return r;
}

#else


#pragma once
#include <complex>
#include <vector>
#include <math.h>
#include <cstdint>

using namespace std;
using REAL = complex<float>;

vector<REAL> fft(vector<REAL> v, int n) {
    if (n == 1) return v;
    vector<REAL> fv(n);
    vector<REAL> ev(n/2), ov(n/2);

    for (int i = 0; i < n/2; i++) ev[i] = v[i] + v[n/2+i];
    auto t1 = fft(ev, n/2);
    for (int i = 0; i < n/2; i++) fv[2*i] = t1[i];

    for (int i = 0; i < n/2; i++) ov[i] = v[i] - v[n/2+i];
    for (int i = 0; i < n/2; i++) ov[i] *= polar(1.0, -2.0*M_PI*i/n);
    auto t2 = fft(ov, n/2);
    for (int i = 0; i < n/2; i++) fv[2*i+1] = t2[i];

    return fv;
}

vector<REAL> rfft(vector<REAL> v, int n) {
    if (n == 1) return v;
    vector<REAL> fv(n);
    vector<REAL> ev(n/2), ov(n/2);

    for (int i = 0; i < n/2; i++) ev[i] = v[i] + v[n/2+i];
    auto t1 = rfft(ev, n/2);
    for (int i = 0; i < n/2; i++) fv[2*i] = t1[i];

    for (int i = 0; i < n/2; i++) ov[i] = v[i] - v[n/2+i];
    for (int i = 0; i < n/2; i++) ov[i] *= polar(1.0, 2.0*M_PI*i/n);
    auto t2 = rfft(ov, n/2);
    for (int i = 0; i < n/2; i++) fv[2*i+1] = t2[i];

    return fv;
}

vector<int> convolve(vector<int> f_, vector<int> g_) {
    int n = 1;
    while (2 * n < f_.size() + g_.size() - 1) n *= 2;
    n *= 2;
    vector<REAL> f(n), g(n);
    for (int i = 0; i < f_.size(); i++) f[i] = f_[i];
    for (int i = 0; i < g_.size(); i++) g[i] = g_[i];

    auto Ff = fft(f, n);
    auto Fg = fft(g, n);
    vector<REAL> Fh(n);
    for (int i = 0; i < n; i++) Fh[i] = Ff[i] * Fg[i];

    auto h = rfft(Fh, n);

    vector<int> ret(n);
    for (int i = 0; i < n; i++) ret[i] = round(h[i].real() / n);

    vector<int> r(f_.size());
    for (int i = 0; i < f_.size(); i++) r[i] = ret[f_.size()/2+i];
    return r;
}


#endif