#include "matrix_shape.hpp"
#include <cassert>
struct C {std::size_t msize,nsize,ksize,max_msize,max_nsize,max_ksize;};
int main(){
    assert(xe::matrix_shape_matches(C{0,16,32,8,0,0},8,16,32));
    assert(!xe::matrix_shape_matches(C{0,16,32,4,0,0},8,16,32));
    assert(!xe::matrix_shape_matches(C{0,16,32,8,0,0},8,8,32));
    assert(!xe::matrix_shape_matches(C{0,16,32,0,0,0},8,16,32));
    assert(xe::matrix_shape_matches(C{8,16,32,0,0,0},8,16,32));
    assert(!xe::matrix_shape_matches(C{8,16,32,0,0,0},0,16,32));
}
