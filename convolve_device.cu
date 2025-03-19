#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <vector>

using namespace std;
using COMP = thrust::complex<double>;

namespace convdev {

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

    vector<int> convolve(vector<int> _f, vector<int> _g) {
        int n = 1;
        while (2 * n < _f.size() + _g.size() - 1) n *= 2;
        n *= 2;
        int blockSize = 32;
        int numBlocks = (n + blockSize - 1) / blockSize;
        int *f_, *g_;
        cudaMalloc(&f_, sizeof(int)*n);
        cudaMalloc(&g_, sizeof(int)*n);
        cudaMemset(f_, 0, sizeof(int) * n);
        cudaMemset(g_, 0, sizeof(int) * n);
        cudaMemcpy(f_, _f.data(), sizeof(int)*_f.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(g_, _g.data(), sizeof(int)*_g.size(), cudaMemcpyHostToDevice);

        COMP *f, *g;
        cudaMalloc(&f, sizeof(COMP)*n);
        cudaMalloc(&g, sizeof(COMP)*n);
        // 配列全体をゼロ初期化

        itor<<<numBlocks, blockSize>>>(f_, f, n);
        itor<<<numBlocks, blockSize>>>(g_, g, n);
        cudaFree(f_); cudaFree(g_);

        fft(f, n);
        fft(g, n);

        kmul<<<numBlocks, blockSize>>>(f, g, n);
        fft(f, n, true);
        kdiv<<<numBlocks, blockSize>>>(f, n, n);

        int* r;
        cudaMalloc(&r, sizeof(int)*n);
        rtoi<<<numBlocks, blockSize>>>(f, r, n);

        vector<int> ret(n);
        cudaMemcpy(ret.data(), r, sizeof(int)*n, cudaMemcpyDeviceToHost);
        cudaFree(f); cudaFree(g); cudaFree(r);
        while (ret.back() == 0) ret.pop_back();
        return ret;
    }

}