#pragma once
#include <algorithm>
#include <complex>
#include <vector>
#include <math.h>
#include <cstdint>

using namespace std;
using REAL = complex<double>;

namespace convhost {
    vector<REAL> fft(vector<REAL> v, int n) {
        if (n == 1) return v;
        vector<REAL> fv(n);
        vector<REAL> ev(n/2), ov(n/2);
        
        for (int i = 0; i < n/2; i++) {
            ev[i] = v[i*2];
            ov[i] = v[i*2+1];
        }
        
        ev = fft(ev, n/2);
        ov = fft(ov, n/2);
        for (int i = 0; i < n/2; i++) {
            ov[i] *= polar(1.0, 2*M_PI * i / n);
            fv[i] = ev[i] + ov[i];
            fv[i+n/2] = ev[i] - ov[i];
        }
    
        return fv;
    }
    
    vector<REAL> rfft(vector<REAL> v, int n) {
        if (n == 1) return v;
        vector<REAL> fv(n);
        vector<REAL> ev(n/2), ov(n/2);
        
        for (int i = 0; i < n/2; i++) {
            ev[i] = v[i*2];
            ov[i] = v[i*2+1];
        }
        
        ev = rfft(ev, n/2);
        ov = rfft(ov, n/2);
        for (int i = 0; i < n/2; i++) {
            ov[i] *= polar(1.0, -2*M_PI * i / n);
            fv[i] = ev[i] + ov[i];
            fv[i+n/2] = ev[i] - ov[i];
        }
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
    
        while (ret.back() == 0) ret.pop_back();
    
        return ret;
    }
}

