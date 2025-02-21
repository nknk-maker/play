#include <BigDecimal.cu>
#include <iostream>
using namespace std;

#define  A  13591409
#define  B  545140134
#define  C  640320

// 1/sqrt(100005)
BigDecimal invrt(BigDecimal a) {
    BigDecimal x(a.siz), _1(a.siz), _5(a.siz);
    x.set(x.siz/2+1, 1); // 0.01
    _1.set(_1.siz/2-1, 1); // 1
    _5.set(_5.siz/2, 5); // 0.5 = 1/2
    for (int k = 1; k < x.siz/2; k <<= 1) {
        x = x + _5 * x * (_1 - a * x * x);
    }
    return x;
}

// binary split
__global__ void ksplit(BigDecimal* x, BigDecimal* y, BigDecimal* z, int siz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int l = 1<<siz;
    int hl = l>>1;
    int pos = i*l;
    if (pos + hl >= n) return;
    BigDecimal nx = x[pos] * x[pos+hl];
    BigDecimal ny = x[pos+hl] * y[pos] + y[pos+hl] * z[pos];
    BigDecimal nz = z[pos] * z[pos+hl];
    x[pos] = nx;
    y[pos] = ny;
    z[pos] = nz;
}

void binary_split(BigDecimal* x, BigDecimal *y, BigDecimal *z, int n) {
    for (int i = 1; (1<<i) <= n; i++) {
        int blockSize = 32;
        int numBlocks = (n>>i + blockSize - 1) / blockSize;
        ksplit<<<numBlocks, blockSize>>>(x, y, z, i, n);
    }
}

__device__ BigDecimal dx(int i) {
    BigDecimal k = i, c = C;
    if (i == 0) {
        BigDecimal ret = 0;
        return ret;
    }
    return k*k*k*c*c*c/24;
}

__device__ BigDecimal dy(int i) {
    BigDecimal k = i;
    return k*B + A;
}

__device__ BigDecimal dz(int i, int n) {
    BigDecimal k = i;
    if (i == n-1) {
        BigDecimal ret = 0;
        return ret;
    }
    return (k*6+1)*(k*2+1)*(k*6+5);
}

__global__ void kx(BigDecimal *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = dx(2*i)+dx(2*i+1);
}

__global__ void ky(BigDecimal *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = dx(2*i+1)*dy(2*i) + dy(2*i+1)*dz(2*i, 2*n);
}

__global__ void kz(BigDecimal *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z[i] = dz(2*i, 2*n)*dz(2*i+1, 2*n);
}


int main() {
    BigDecimal p = 10;
    cout << p << endl;
    int n = 1<<8;
    BigDecimal *x, *y, *z;
    cudaMalloc(&x, sizeof(BigDecimal)*n);
    cudaMalloc(&y, sizeof(BigDecimal)*n);
    cudaMalloc(&z, sizeof(BigDecimal)*n);
    int blockSize = 32;
    int numBlocks = (n + blockSize - 1) / blockSize;
    kx<<<numBlocks, blockSize>>>(x, n);
    ky<<<numBlocks, blockSize>>>(y, n);
    kz<<<numBlocks, blockSize>>>(z, n);
    binary_split(x, y, z, n);
    BigDecimal X, Y;
    cout << X.siz << endl;
    cudaMemcpy(&X, x, sizeof(BigDecimal), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Y, y, sizeof(BigDecimal), cudaMemcpyDeviceToHost);
    BigDecimal a;
    a.set(a.siz/2+2, 4);
    a.set(a.siz/2+3, 4);
    a.set(a.siz/2+4, 3);
    a.set(a.siz/2+5, 9);
    a.set(a.siz/2+7, 7);
    a.set(a.siz/2+8, 2);
    a.set(a.siz/2+9, 4);
    cout << X.siz << ' ' << a.siz << endl;
    X * a * invrt(10005) / Y;
}


