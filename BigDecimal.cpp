#include <vector>
#include <iostream>
#include <assert.h>
#include <convolve.cpp>
#include <chrono>
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
    dcm(const dcm& b) : v(b.v) {}
    const dcm val() {
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
    dcm& operator=(const dcm& b) {
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
    const dcm operator+(const int& rhs) const {
        return *this + dcm(rhs);
    }
    const dcm operator-(const int& rhs) const {
        return *this - dcm(rhs);
    }
    const dcm operator*(const int& rhs) const {
        return *this * dcm(rhs);
    }
    const dcm operator/(const int& rhs) const {
        return *this / dcm(rhs);
    }
    friend dcm operator+(int lhs, const dcm& rhs) {
        return dcm(lhs) + rhs;
    }
    friend dcm operator-(int lhs, const dcm& rhs) {
        return dcm(lhs) - rhs;
    }
    friend dcm operator*(int lhs, const dcm& rhs) {
        return dcm(lhs) * rhs;
    }
    friend dcm operator/(int lhs, const dcm& rhs) {
        return dcm(lhs) / rhs;
    }
    const dcm operator+(const dcm& rhs) const {
        dcm ret = *this;
        return ret += rhs;
    }
    const dcm operator-(const dcm& rhs) const {
        dcm ret = *this;
        return ret -= rhs;
    }
    const dcm operator*(const dcm& rhs) const {
        dcm ret = *this;
        return ret *= rhs;
    }
    const dcm operator/(const dcm& rhs) const {
        dcm ret = *this;
        return ret /= rhs;
    }
    const dcm &operator+=(const dcm& b) {
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
    const dcm &operator-=(const dcm& b) {
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
    const dcm &operator*=(const dcm& b) {
        v = convolve(v, b.v);
        val();
        sgn ^= b.sgn;
        return *this;
    }
    const dcm &operator/=(dcm b) {
        cout << -1 << endl;
        dcm p = 1, q = *this;
        int need_dg = v.size() - b.v.size() + 1; // 必要桁数
        // 1桁目を一致させる
        for (int i = 0; i < 5; i++) {
            vector<int> tmp(b.v.size()*(1ll<<i)+1);
            tmp.back() = 2;
            dcm two = tmp;
            p = p * (two - b * p);
        }
        // 1桁のみにする
        p.val();
        p.v[0] = p.v.back();
        p.v.resize(1);
        // *this以上の桁数まで合わせる
        for (int i = 0; (1ll<<i) < need_dg; i++) {
            cout << p << endl;
            vector<int> tmp(b.v.size()+p.v.size()); 
            tmp.back() = 2;
            dcm two = tmp;
            p = p * (two - b * p);
            move(p.v.end()-(1ll<<i+1), p.v.end(), p.v.begin());
            p.v.resize(1ll<<i+1);
        }
        int shift = b.v.size()+p.v.size()-1;
        cout << p.v.size() << ' ' << shift << endl;
        *this *= p;
        move(v.begin()+shift, v.end(), v.begin()); // シフトされた分を戻す
        v.resize(v.size()-shift);
        // 答えが一致するか確認し、一致しない場合は1加算
        if (q.v != (b**this).v) v[0]++;
        val();
        sgn ^= b.sgn;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const dcm& b) {
        for (int i = b.v.size()-1; i >= 0; i--) os << b.v[i];
        return os;
    }
};

dcm invrt(dcm b, int digits) { // 1/√b を10^digits倍したものを返す
    dcm p = 1;
    // 1桁目を一致させる
    for (int i = 0; i <= 6; i++) {
        vector<int> tmp((b.v.size()/2+b.v.size()%2+p.v.size())*2-1); 
        tmp.back() = 3;
        dcm three = tmp;
        p = p * (three - b * p * p) / 2;
        move(p.v.end()-(1ll<<i+1), p.v.end(), p.v.begin());
        p.v.resize(1ll<<i+1);
    }
    // 1桁のみにする
    p.val();
    p.v[0] = p.v.back();
    p.v.resize(1);
    int shift;
    // *this以上の桁数まで合わせる
    for (int i = 0; (1ll<<i) <= digits*2; i++) {
        vector<int> tmp((b.v.size()/2+b.v.size()%2+p.v.size())*2-1); 
        tmp.back() = 3;
        dcm three = tmp;
        p = p * (three - b * p * p) / 2;;
        move(p.v.end()-(1ll<<i+1), p.v.end(), p.v.begin());
        p.v.resize(1ll<<i+1);
        shift = 1ll<<i+1;
    }
    move(p.v.begin()+shift-digits, p.v.end(), p.v.begin());
    p.v.resize(p.v.size()-shift+digits);
    return p;
}