#include <vector>
#include <iostream>
#include <assert.h>
#include <convolve_host.cu>
#include <convolve_device.cu>
using namespace std;

struct dcm {
    vector<int> v;
    bool sgn = false;
    dcm() {}
    dcm(const int& b) {
        int p = 0, x = b;
        while (x > 0) {
            v.push_back(x%10);
            x /= 10;
            p++;
        }
    }
    dcm(const long long& b) {
        int p = 0; long long x = b;
        while (x > 0) {
            v.push_back(x%10);
            x /= 10;
            p++;
        }
    }
    dcm(const vector<int>& b) : v(b) {}
    dcm(const dcm& b) : v(b.v), sgn(b.sgn) {}
    dcm val() {
        for (int i = 0; i < v.size()-1; i++) {
            int p = v[i] / 10;
            if (v[i] < 0) p--;
            v[i+1] += p;
            v[i] -= p * 10;
        }
        while (v.back() > 9) {
            int p = v.back() / 10;
            v.back() -= p * 10;
            v.push_back(p);
        }
        return *this;
    }
    void set(int i, int a) {
        v[i] = a;
    }
    dcm& operator<<=(int b) {
        v.resize(v.size()+b); 
        move(v.begin(), v.begin()+v.size()-b, v.begin()+b); 
        fill(v.begin(), v.begin()+b, 0);
        return (*this);
    }
    dcm& operator>>=(int b) {
        v = vector<int>(v.begin()+b, v.end());
        return (*this);
    }
    dcm operator<<(int b) {return dcm(*this) <<= b;}
    dcm operator>>(int b) {return dcm(*this) >>= b;}
    dcm& operator=(const dcm& b) {
        sgn = b.sgn;
        v = b.v;
        return *this;
    }
    dcm& operator=(const int& b) {
        v.clear();
        int p = 0, x = b;
        while (x > 0) {
            v.push_back(x%10);
            x /= 10;
            p++;
        }
        return *this;
    }
    dcm& operator=(const long long& b) {
        v.clear();
        int p = 0; long long x = b;
        while (x > 0) {
            v.push_back(x%10);
            x /= 10;
            p++;
        }
        return *this;
    }
    bool operator==(const dcm& b) const {return v == b.v;}
    bool operator!=(const dcm& b) const {return v != b.v;}
    dcm operator+(const long long& rhs) const {return *this + dcm(rhs);}
    dcm operator-(const long long& rhs) const {return *this - dcm(rhs);}
    dcm operator*(const long long& rhs) const {return *this * dcm(rhs);}
    dcm operator/(const long long& rhs) const {return *this / dcm(rhs);}
    friend dcm operator+(long long lhs, const dcm& rhs) {return dcm(lhs) + rhs;}
    friend dcm operator-(long long lhs, const dcm& rhs) {return dcm(lhs) - rhs;}
    friend dcm operator*(long long lhs, const dcm& rhs) {return dcm(lhs) * rhs;}
    friend dcm operator/(long long lhs, const dcm& rhs) {return dcm(lhs) / rhs;}
    dcm operator+(const dcm& rhs) const {return dcm(*this) += rhs;}
    dcm operator-(const dcm& rhs) const {return dcm(*this) -= rhs;}
    dcm operator*(const dcm& rhs) const {return dcm(*this) *= rhs;}
    dcm operator/(const dcm& rhs) const {return dcm(*this) /= rhs;}
    dcm operator+=(const dcm& b) {
        if (sgn != b.sgn) {
            dcm nb = b;
            nb.sgn ^= 1;
            return *this -= nb;
        }
        int Min = min(v.size(), b.v.size()), Max = max(v.size(), b.v.size());
        v.resize(Max);
        for (int i = 0; i < b.v.size(); i++) v[i] += b.v[i];
        val();
        return (*this);
    }
    dcm operator-=(const dcm& b) {
        if (sgn != b.sgn) {
            dcm nb = b;
            nb.sgn ^= 1;
            return *this += nb;
        }
        int Min = min(v.size(), b.v.size()), Max = max(v.size(), b.v.size());
        v.resize(Max);
        for (int i = 0; i < b.v.size(); i++) v[i] -= b.v[i];
        val();    
        return (*this);
    }
    const dcm operator*=(const dcm& b) {
        if (v.size()+b.v.size() <= 50000) v = convhost::convolve(v, b.v);
        else v = convdev::convolve(v, b.v);
        val();
        sgn ^= b.sgn;
        return *this;
    }
    const dcm operator/=(dcm b) {
        dcm p, np = 1, q = *this;
        int req_d = v.size() - b.v.size() + 1; // 必要桁数
        int d = min(req_d, 3), bd = min((int)b.v.size(), 6);
        np <<= d;
        // 1桁目を一致させる
        while (p != np) {
            p = np;
            dcm rb = b >> b.v.size() - bd;
            np = p * ((dcm(2)<<d+bd)-rb*p);
            np >>= d+bd;
        }
        if (d < req_d) {
            p.v.clear();
            while (p != np) {
                p = np;
                dcm rb = b >> b.v.size() - bd;
                np = p * ((dcm(2)<<d+bd)-rb*p);
                np >>= d+bd;
                int nd = min(d*2, req_d);
                if (nd > d) np <<= nd - d;
                d = nd;
                bd = min(bd*2, (int)b.v.size());
            }
        }
        int shift_cnt = v.size();
        *this = ((*this) * p);
        *this >>= shift_cnt+1;
        // 答えが一致するか確認し、一致しない場合は1加算
        if (q != (b**this)) v[0]++;
        val();
        sgn ^= b.sgn;
        return *this;
    }
    dcm half() {return dcm(*this*5)>>1;}
    friend std::ostream& operator<<(std::ostream& os, const dcm& b) {
        if (b.sgn) os << "-";
        for (int i = b.v.size()-1; i >= 0; i--) os << b.v[i];
        return os;
    }
};

dcm invrt(dcm b, int digits) { // 1/√b を10^digits倍したものを返す
    dcm p, np = 1;
    int d = min(digits, 3), bd = min((int)b.v.size(), 6);
    np <<= d;
    while (p != np) {
        p = np;
        dcm rb = b >> b.v.size() - bd;
        np = (p*((dcm(3)<<(bd/2+bd%2+d)*2) - rb * p * p)).half();
        np >>= (bd/2+bd%2+d)*2;
    }
    if (d < digits) {
        p.v.clear();
        while (p != np) {
            p = np;
            dcm rb = b >> b.v.size() - bd;
            np = (p*((dcm(3)<<(bd/2+bd%2+d)*2) - rb * p * p)).half();
            np >>= (bd/2+bd%2+d)*2;
            int nd = min(d*2, digits);
            if (nd > d) np <<= nd - d;
            d = nd;
            bd = min(bd*2, (int)b.v.size());
        }
    }
    return p;
}