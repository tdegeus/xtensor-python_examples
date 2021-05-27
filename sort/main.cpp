#include <xtensor.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

template <class T>
inline auto foo(const T& arg)
{
    return xt::sort(arg);
}

PYBIND11_MODULE(mymodule, m)
{
    xt::import_numpy();
    m.doc() = "Module description";
    m.def("foo", &foo<xt::pytensor<size_t, 1>>, "Function description", py::arg("arg"));
}
