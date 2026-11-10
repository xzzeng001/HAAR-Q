#include <pybind11/complex.h>
#include <torch/extension.h>

namespace haar_q_hamiltonian {

// `prepare` parses a Python dictionary of Hamiltonian terms and converts it
// into a structured tuple of tensors. The tuple is stored on the Python side
// and passed to subsequent PyTorch operator calls.
//
// Each key-value pair in the input dictionary `hamiltonian` represents one term.
// A key is a tuple of operator tuples, each containing two elements:
// - An integer site index identifying where the operator acts.
// - An integer operator type (0 for annihilation, 1 for creation).
// The value is a real or complex floating-point coefficient for the term.
//
// The function constructs three tensors from the dictionary:
// - `site`: int16, shape [term_number, max_op_number], storing operator site indices.
// - `kind`: uint8, shape [term_number, max_op_number], storing operator types.
//   Type codes are:
//   - 0: annihilation operator
//   - 1: creation operator
//   - 2: padding (identity operator)
// - `coef`: float64, shape [term_number, 2], storing real and imaginary coefficient parts.
//
// The template parameter `max_op_number` limits the number of operators per term;
// it is typically 4 for two-body interactions.
template<std::int64_t max_op_number>
auto prepare(py::dict hamiltonian) {
    std::int64_t term_number = hamiltonian.size();

    auto site = torch::empty({term_number, max_op_number}, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
    // No initialization is required.
    auto kind = torch::full({term_number, max_op_number}, 2, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    // Initialize padding entries to 2 (the identity operator).
    auto coef = torch::empty({term_number, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    // No initialization is required.

    auto site_accessor = site.accessor<std::int16_t, 2>();
    auto kind_accessor = kind.accessor<std::uint8_t, 2>();
    auto coef_accessor = coef.accessor<double, 2>();

    std::int64_t index = 0;
    for (auto& item : hamiltonian) {
        auto key = item.first.cast<py::tuple>();
        auto value_is_float = py::isinstance<py::float_>(item.second);
        auto value = value_is_float ? std::complex<double>(item.second.cast<double>()) : item.second.cast<std::complex<double>>();

        std::int64_t op_number = key.size();
        for (std::int64_t i = 0; i < op_number; ++i) {
            auto tuple = key[i].cast<py::tuple>();
            site_accessor[index][i] = tuple[0].cast<std::int16_t>();
            kind_accessor[index][i] = tuple[1].cast<std::uint8_t>();
        }

        coef_accessor[index][0] = value.real();
        coef_accessor[index][1] = value.imag();

        ++index;
    }

    return std::make_tuple(site, kind, coef);
}

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif

#if N_QUBYTES == 0
// Expose `prepare` to Python.
PYBIND11_MODULE(haar_q_hamiltonian, m) {
    m.def("prepare", prepare</*max_op_number=*/4>, py::arg("hamiltonian"));
}
#endif

#if N_QUBYTES != 0
#define HAAR_Q_LIBRARY_HELPER(x, y) haar_q_hamiltonian_##x##_##y
#define HAAR_Q_LIBRARY(x, y) HAAR_Q_LIBRARY_HELPER(x, y)
TORCH_LIBRARY_FRAGMENT(HAAR_Q_LIBRARY(N_QUBYTES, PARTICLE_CUT), m) {
    m.def("apply_within(Tensor configs_i, Tensor psi_i, Tensor configs_j, Tensor site, Tensor kind, Tensor coef) -> Tensor");
    m.def("find_relative(Tensor configs_i, Tensor psi_i, int count_selected, Tensor site, Tensor kind, Tensor coef, Tensor configs_exclude) -> Tensor"
    );
    m.def("relative_candidate_pool(Tensor configs_i, Tensor psi_i, int count_selected, Tensor site, Tensor kind, Tensor coef, Tensor configs_exclude) -> Tensor");
    m.def("list_relative(Tensor configs_i, Tensor psi_i, Tensor site, Tensor kind, Tensor coef, Tensor configs_exclude) -> (Tensor, Tensor)");
    m.def("diagonal_term(Tensor configs, Tensor site, Tensor kind, Tensor coef) -> Tensor");
}
#undef HAAR_Q_LIBRARY
#undef HAAR_Q_LIBRARY_HELPER
#endif

} // namespace haar_q_hamiltonian
