#include <BigDecimal.cpp>
#include <iostream>
using namespace std;

#define C 64320
#define C3_OVER24 11087327232000LL

struct PQT {
    dcm p, q, t;
    PQT(dcm a, dcm b, dcm c) : p(a), q(b), t(c) {}
};

dcm chudnovsky_bs(int digits) {
    auto bs = [&](auto f, int a, int b)->PQT {
        dcm Pab, Qab, Tab;
        if (b - a == 1) {
            if (a == 0) Pab = Qab = 1;
            else {
                Pab = (6*a-5)*(2*a-1)*(6*a-1);
                Qab = a*a*a*C3_OVER24;
            }
            Tab = Pab * (13591409 + 545140134*a);
            if (a & 1) Tab.sgn = true;
        }
        else {
            int m = (a + b) / 2;
            auto [Pam, Qam, Tam] = f(f, a, m);
            auto [Pmb, Qmb, Tmb] = f(f, m, b);
            Pab = Pam * Pmb;
            Qab = Qam * Qmb;
            Tam = Qmb * Tam + Pam * Tmb;
        }
        return PQT(Pab, Qab, Tab);
    };

    double Digits_per_term = log10(C3_OVER24/72);
    int n = digits / Digits_per_term + 1;
    cout << n << endl;

    PQT Z = bs(bs, 0, n);
    vector<int> tmp(digits+1);
    tmp.back() = 1;
    dcm A(4270934400LL);
    return A*Z.q*invrt(10005, digits);
}

int main() {
    cout << chudnovsky_bs(20) << endl;
}