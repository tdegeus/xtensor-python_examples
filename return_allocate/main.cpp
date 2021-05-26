#include <xtensor.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

template <size_t RANK, template <class T, size_t N, class... Args> class X, class T, size_t N, class... Args>
auto allocate_tensor(const std::array<size_t, RANK>& shape, X<T, N, Args...>&)
{
     return X<T, RANK, Args...>(shape);
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
    m.def("foo", &foo, "Function description", py::arg("arg"))
}
