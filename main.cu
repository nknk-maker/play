#include <BigDecimal.cu>
#include <iostream>
using namespace std;

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

void binary_split(BigDecimal* x, BigDecimal *y, BigDecimal *z, int n) {
    for (int i = 1; (1<<i) <= n; i++) {
        int blockSize = 32;
        int numBlocks = (n>>i + blockSize - 1) / blockSize;
        for (int j = 0; j < n; j += (1<<i)) {
            int k = i>>1;
            BigDecimal nx = x[j] * x[j+k];
            BigDecimal ny = x[j+k] * y[j] + y[j+k] * z[j];
            BigDecimal nz = z[j] * z[j+k];
            x[j] = nx;
            y[j] = ny;
            z[j] = nz;
        }
    }
}



int main() {
    int n = 1<<10;
    BigDecimal A = 13591409, B = 545140134, C = 640320;
    BigDecimal x[n], y[n], z[n];
    for (int i = 0; i < n; i++) {
        BigDecimal k = i, kk = i+1;
        cout << k << endl;
        BigDecimal x1 = k*k*k*C*C*C/24, x2 = kk*kk*kk*C*C*C/24, y1 = A+B*k, y2 = A+B*kk, z1 = (k*6+1)*(k*2+1)*(k*6+5), z2 = (kk*6+1)*(kk*2+1)*(kk*6+5);
        if (i == 0) x1 = 1;
        if (i == n-1) z2 = 0;
        x[i] = x1*x2;
        y[i] = x2*y1+y2*z1;
        z[i] = z1*z2;
    }
    binary_split(x, y, z, n);
    BigDecimal a = 4270934400ll;
    cout << a*x[0]*invrt(10005)/y[0] << endl;
}


