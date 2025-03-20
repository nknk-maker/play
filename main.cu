#include <BigDecimal.cu>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std;

const dcm a(13591409LL), B(545140134), C(640320LL), C3_OVER24(10939058860032000LL);

struct PQT {
    dcm p, q, t;
    PQT(dcm a, dcm b, dcm c) : p(a), q(b), t(c) {}
};

dcm chudnovsky_bs(int digits) {
    auto bs = [&](auto f, int a, int b)->PQT {
        dcm Pab, Qab, Tab, aa = a;
            if (b - a == 1) {
            if (a == 0) Pab = Qab = 1;
            else {
                Pab = (6*aa-5)*(2*aa-1)*(6*aa-1);
                Qab = aa*aa*aa*C3_OVER24;
            }
            Tab = Pab * (13591409LL + 545140134LL*a);
            if (a & 1) Tab.sgn = true;
        }
        else {
            int m = (a + b) / 2;
            auto [Pam, Qam, Tam] = f(f, a, m);
            auto [Pmb, Qmb, Tmb] = f(f, m, b);
            Pab = Pam * Pmb;
            Qab = Qam * Qmb;
            Tab = Qmb * Tam + Pam * Tmb;
        }
        return PQT(Pab, Qab, Tab);
    };
    double Digits_per_term = log10((double)10939058860032000LL/72);
    int n = digits / Digits_per_term + 1;
    PQT Z = bs(bs, 0, n);
    dcm A(4270934400LL);
    return A*Z.q*invrt(10005, digits)/Z.t;
}

int main() {
    cin.tie(nullptr);
    ios_base::sync_with_stdio(false);
    auto start = chrono::high_resolution_clock::now();
    ofstream file("pi_out.txt");
    file << chudnovsky_bs(10000000) << endl;
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "実行時間: " << duration.count() << " ms" << endl;
}