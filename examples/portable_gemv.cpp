// Portable baseline: no fixed work-group or subgroup width, independent host int64 oracle.
#include "bench.hpp"
#include <cstdint>
#include <iostream>
#include <random>
int main() {
    auto q = bench::make_queue();
    if (!q.get_device().has(sycl::aspect::usm_device_allocations)) {
        std::cout << "SKIP: device USM unavailable\n"; return 77;
    }
    std::mt19937 rng(17);
    std::uniform_int_distribution<int> dist(-127,127);
    for (int k : {1, 31, 32, 33, 257}) for (int rows : {1, 7, 17}) {
        std::vector<int8_t> w(size_t(k)*rows), x(k);
        for (auto& v:w) v=static_cast<int8_t>(dist(rng));
        for (auto& v:x) v=static_cast<int8_t>(dist(rng));
        std::vector<int32_t> got(rows);
        auto dw=sycl::malloc_device<int8_t>(w.size(),q);
        auto dx=sycl::malloc_device<int8_t>(x.size(),q);
        auto dy=sycl::malloc_device<int32_t>(rows,q);
        if (!dw || !dx || !dy) throw std::bad_alloc();
        q.memcpy(dw,w.data(),w.size()); q.memcpy(dx,x.data(),x.size());
        q.parallel_for(sycl::range<1>(rows),[=](sycl::id<1> id){
            int32_t acc=0;
            for(int i=0;i<k;++i) acc+=int32_t(dw[size_t(id[0])*k+i])*int32_t(dx[i]);
            dy[id[0]]=acc;
        });
        q.memcpy(got.data(),dy,rows*sizeof(int32_t)).wait_and_throw();
        bool ok=true;
        for(int r=0;r<rows;++r) {
            int64_t expected=0;
            for(int i=0;i<k;++i) expected+=int64_t(w[size_t(r)*k+i])*int64_t(x[i]);
            ok &= got[r]==expected;
        }
        sycl::free(dw,q); sycl::free(dx,q); sycl::free(dy,q);
        if (!ok) {std::cerr<<"FAIL: "<<rows<<" x "<<k<<"\n"; return 1;}
    }
    std::cout<<"PASS: portable int8 GEMV, 15 shapes, exact int64 host oracle\n";
}
