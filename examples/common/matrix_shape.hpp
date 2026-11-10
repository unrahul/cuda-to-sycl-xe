#pragma once
#include <cstddef>
namespace xe {
// Runtime combinations may mix bounded dimensions with fixed ones (e.g. M<=8,N=16,K=32).
inline bool matrix_dim_matches(std::size_t fixed, std::size_t maximum, std::size_t requested) {
    return requested > 0 && (fixed ? requested == fixed : maximum && requested <= maximum);
}
template<class Combination>
bool matrix_shape_matches(const Combination& c, std::size_t m, std::size_t n, std::size_t k) {
    return matrix_dim_matches(c.msize,c.max_msize,m) &&
           matrix_dim_matches(c.nsize,c.max_nsize,n) &&
           matrix_dim_matches(c.ksize,c.max_ksize,k);
}
}
