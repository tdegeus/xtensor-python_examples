#include <xtensor.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

template <size_t RANK, class T>
struct allocate
{
    using type = typename std::array<T, RANK>;
};

template <size_t RANK, class EC, size_t N, xt::layout_type L, class Tag>
struct allocate<RANK, xt::xtensor<EC, N, L, Tag>>
{
    using type = typename xt::xtensor<EC, RANK, L, Tag>;
};

#ifdef PY_TENSOR_HPP
template <size_t RANK, class EC, size_t N, xt::layout_type L>
struct allocate<RANK, xt::pytensor<EC, N, L>>
{
    using type = typename xt::pytensor<EC, RANK, L>;
};
#endif

template <size_t RANK, template <class EC, size_t N, xt::layout_type> class X, class EC, size_t N, xt::layout_type L>
auto free_allocate_tensor(const std::array<size_t, RANK>& shape, const X<EC, N, L>&) -> X<EC, RANK, L>
{
     return X<EC, RANK, L>::from_shape(shape);
}

template <class T>
inline auto foo(const T& arg)
{
    std::array<size_t, 2> shape = {3, 4};
    auto ret = free_allocate_tensor(shape, arg);
    ret.fill(5);
    return ret;
}

template <class T>
inline auto bar(const T& arg) -> typename allocate<2, T>::type
{
    using return_type = typename allocate<2, T>::type;
    std::array<size_t, 2> shape = {3, 4};
    return_type ret = return_type::from_shape(shape);
    ret.fill(5);
    return ret;
}


PYBIND11_MODULE(mymodule, m)
{
    xt::import_numpy();
    m.doc() = "Module description";
    m.def("foo",
          static_cast<xt::pytensor<size_t, 2> (*)(const xt::pytensor<size_t, 1>&)>(&foo),
          "Function description",
          py::arg("arg"));

    m.def("bar",
          &bar<xt::pytensor<size_t, 1>>,
          "Function description",
          py::arg("arg"));
}
