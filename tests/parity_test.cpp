#include "bench.hpp"
#include <cassert>
#include <limits>
int main() {
    using bench::compare;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    assert(compare(std::vector<float>{1}, std::vector<float>{1}).bitwise);
    assert(std::isinf(compare(std::vector<float>{}, std::vector<float>{1}).rel_err));
    assert(std::isinf(compare(std::vector<float>{nan}, std::vector<float>{nan}).rel_err));
    assert(std::isinf(compare(std::vector<float>{1}, std::vector<float>{inf}).rel_err));
    assert(std::isinf(compare(std::vector<float>{1}, std::vector<double>{nan}).rel_err));
    assert(!bench::within_tolerance(std::vector<float>{1e6f, 0.01f}, std::vector<double>{1e6, 0}, 1e-5, 1e-5));
    assert(!bench::within_tolerance(std::vector<float>{nan}, std::vector<double>{0}, 1e-5, 1e-5));
    assert(bench::within_tolerance(std::vector<float>{1.000001f}, std::vector<double>{1}, 1e-5, 1e-5));
    assert(compare(std::vector<float>{2}, std::vector<double>{1}).rel_err == 1);
}
