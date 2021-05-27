#include <xtensor.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

// template <size_t RANK, template <class EC, size_t N, xt::layout_type, class> class X, class EC, size_t N, xt::layout_type L, class Tag>
// auto allocate_tensor(const std::array<size_t, RANK>& shape, X<EC, N, L, Tag>&)
// {
//      return X<EC, RANK, L, Tag>(shape);
// }

template <size_t RANK, template <class EC, size_t N, xt::layout_type> class X, class EC, size_t N, xt::layout_type L>
auto allocate_tensor(const std::array<size_t, RANK>& shape, const X<EC, N, L>&)
{
     return X<EC, RANK, L>(shape);
}

template <class T>
inline auto foo(const T& arg)
{
    std::array<size_t, 2> shape = {3, 4};
    auto ret = allocate_tensor(shape, arg);
    ret.fill(5);
    return ret;
}


PYBIND11_MODULE(mymodule, m)
{
    xt::import_numpy();
    m.doc() = "Module description";
    m.def("foo", &foo<xt::pytensor<size_t, 1>>, "Function description", py::arg("arg"));
}
