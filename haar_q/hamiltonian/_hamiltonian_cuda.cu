#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <thrust/sort.h>
#include <torch/extension.h>

namespace haar_q_hamiltonian_cuda {

constexpr torch::DeviceType device = torch::kCUDA;

// Assign more lanes to terms when only a few determinants are present. In a
// large basis, each warp processes a row of terms for one determinant.
std::pair<dim3, dim3> term_batch_launch(std::int64_t terms, std::int64_t batch, int device_id) {
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    unsigned int y = 1;
    while (y < 8 && y < batch) {
        y *= 2;
    }
    unsigned int x = 256 / y;
    while (x > 1 && terms <= x / 2 && y < batch) {
        x /= 2;
        y *= 2;
    }
    const auto needed_x = terms ? 1 + (terms - 1) / x : 1;
    const auto needed_y = batch ? 1 + (batch - 1) / y : 1;
    TORCH_CHECK(needed_x <= prop.maxGridSize[0], "Hamiltonian term count exceeds the CUDA x-grid limit.");
    const auto blocks_y = std::min<std::int64_t>(needed_y, prop.maxGridSize[1]);
    return {dim3(static_cast<unsigned int>(needed_x), static_cast<unsigned int>(blocks_y)), dim3(x, y)};
}

// Configuration data is bit-packed (8 sites per byte).
// Lexicographic comparison of the packed byte arrays defines the configuration ordering.
// This comparator is used by thrust::sort and subsequent binary searches.
template<typename T, std::int64_t size>
struct array_less {
    __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        for (std::int64_t i = 0; i < size; ++i) {
            if (lhs[i] < rhs[i]) {
                return true;
            }
            if (lhs[i] > rhs[i]) {
                return false;
            }
        }
        return false;
    }
};

// Compute a configuration's probability weight (the squared amplitude magnitude).
// Top-K selection uses this weight to rank important configurations.
template<typename T, std::int64_t size>
struct array_square_greater {
    __device__ T square(const std::array<T, size>& value) const {
        T result = 0;
        for (std::int64_t i = 0; i < size; ++i) {
            result += value[i] * value[i];
        }
        return result;
    }
    __device__ bool operator()(const std::array<T, size>& lhs, const std::array<T, size>& rhs) const {
        return square(lhs) > square(rhs);
    }
};

// Locate a site using its byte offset (index/8) and bit position (index%8).
__device__ bool get_bit(std::uint8_t* data, std::uint8_t index) {
    return ((*data) >> index) & 1;
}

__device__ void set_bit(std::uint8_t* data, std::uint8_t index, bool value) {
    if (value) {
        *data |= (1 << index);
    } else {
        *data &= ~(1 << index);
    }
}

// Apply one Hamiltonian term to the current configuration.
// 1. Check that fermionic creation/annihilation operations obey Pauli exclusion.
// 2. Flip bits in current_configs in place to avoid copying.
// 3. Account for the fermionic exchange sign (-1)^N using the particle parity before the target site.
template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ std::pair<bool, bool> hamiltonian_apply_kernel(
    std::array<std::uint8_t, n_qubytes>& current_configs,
    std::int64_t term_index,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind // term_number
) {
    static_assert(particle_cut == 1 || particle_cut == 2, "particle_cut != 1 or 2 not implemented");
    bool success = true;
    bool parity = false;
    for (std::int64_t op_index = max_op_number; op_index-- > 0;) {
        std::int16_t site_single = site[term_index][op_index];
        std::uint8_t kind_single = kind[term_index][op_index];
        if (kind_single == 2) {
            continue;
        }
        std::uint8_t to_what = kind_single;
        if (get_bit(&current_configs[site_single / 8], site_single % 8) == to_what) {
            success = false;
            break;
        }
        set_bit(&current_configs[site_single / 8], site_single % 8, to_what);
        if constexpr (particle_cut == 1) {
            for (std::int16_t s = 0; s < site_single; ++s) {
                parity ^= get_bit(&current_configs[s / 8], s % 8);
            }
        }
    }
    return std::make_pair(success, parity);
}

// Compute the projection of H|psi> onto the specified basis (result_configs).
// Each thread finds the target index by binary search in the sorted result_configs.
// Multiple source states can reach the same target, so atomicAdd accumulates amplitudes safely.
template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ void apply_within_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t result_batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* result_configs, // result_batch_size
    std::array<double, 2>* result_psi
) {
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(
        /*current_configs=*/current_configs,
        /*term_index=*/term_index,
        /*site=*/site,
        /*kind=*/kind
    );

    if (!success) {
        return;
    }
    success = false;
    std::int64_t low = 0;
    std::int64_t high = result_batch_size - 1;
    std::int64_t mid = 0;
    auto less = array_less<std::uint8_t, n_qubytes>();
    while (low <= high) {
        mid = (low + high) / 2;
        if (less(current_configs, result_configs[mid])) {
            high = mid - 1;
        } else if (less(result_configs[mid], current_configs)) {
            low = mid + 1;
        } else {
            success = true;
            break;
        }
    }
    if (!success) {
        return;
    }
    std::int8_t sign = parity ? -1 : +1;
    atomicAdd(&result_psi[mid][0], sign * (coef[term_index][0] * psi[batch_index][0] - coef[term_index][1] * psi[batch_index][1]));
    atomicAdd(&result_psi[mid][1], sign * (coef[term_index][0] * psi[batch_index][1] + coef[term_index][1] * psi[batch_index][0]));
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void apply_within_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t result_batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* result_configs, // result_batch_size
    std::array<double, 2>* result_psi
) {
    const std::int64_t term_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (term_index >= term_number) {
        return;
    }
    const auto batch_stride = static_cast<std::int64_t>(gridDim.y) * blockDim.y;
    for (std::int64_t batch_index = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         batch_index < batch_size; batch_index += batch_stride) {
        apply_within_kernel<max_op_number, n_qubytes, particle_cut>(
            /*term_index=*/term_index,
            /*batch_index=*/batch_index,
            /*term_number=*/term_number,
            /*batch_size=*/batch_size,
            /*result_batch_size=*/result_batch_size,
            /*site=*/site,
            /*kind=*/kind,
            /*coef=*/coef,
            /*configs=*/configs,
            /*psi=*/psi,
            /*result_configs=*/result_configs,
            /*result_psi=*/result_psi
        );
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto apply_within_interface(
    const torch::Tensor& configs,
    const torch::Tensor& psi,
    const torch::Tensor& result_configs,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef
) -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t result_batch_size = result_configs.size(0);
    std::int64_t term_number = site.size(0);

    at::cuda::CUDAGuard cuda_device_guard(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);

    TORCH_CHECK(configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(psi.device().type() == torch::kCUDA, "psi must be on CUDA.")
    TORCH_CHECK(psi.device().index() == device_id, "psi must be on the same device as others.");
    TORCH_CHECK(psi.is_contiguous(), "psi must be contiguous.")
    TORCH_CHECK(psi.dtype() == torch::kFloat64, "psi must be float64.")
    TORCH_CHECK(psi.dim() == 2, "psi must be 2D.")
    TORCH_CHECK(psi.size(0) == batch_size, "psi batch size must match the provided batch_size.");
    TORCH_CHECK(psi.size(1) == 2, "psi must contain 2 elements for each batch.");

    TORCH_CHECK(result_configs.device().type() == torch::kCUDA, "result_configs must be on CUDA.")
    TORCH_CHECK(result_configs.device().index() == device_id, "result_configs must be on the same device as others.");
    TORCH_CHECK(result_configs.is_contiguous(), "result_configs must be contiguous.")
    TORCH_CHECK(result_configs.dtype() == torch::kUInt8, "result_configs must be uint8.")
    TORCH_CHECK(result_configs.dim() == 2, "result_configs must be 2D.")
    TORCH_CHECK(result_configs.size(0) == result_batch_size, "result_configs batch size must match the provided result_batch_size.")
    TORCH_CHECK(result_configs.size(1) == n_qubytes, "result_configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(site.device().type() == torch::kCUDA, "site must be on CUDA.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCUDA, "kind must be on CUDA.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCUDA, "coef must be on CUDA.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    auto sorted_result_configs = result_configs.clone(torch::MemoryFormat::Contiguous);
    auto result_sort_index = torch::arange(result_batch_size, torch::TensorOptions().dtype(torch::kInt64).device(device, device_id));
    thrust::sort_by_key(
        policy,
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_result_configs.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_result_configs.data_ptr()) + result_batch_size,
        reinterpret_cast<std::int64_t*>(result_sort_index.data_ptr()),
        array_less<std::uint8_t, n_qubytes>()
    );
    auto sorted_result_psi = torch::zeros({result_batch_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    const auto [num_blocks, threads_per_block] = term_batch_launch(term_number, batch_size, device_id);
    apply_within_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks, threads_per_block, 0, stream>>>(
        /*term_number=*/term_number,
        /*batch_size=*/batch_size,
        /*result_batch_size=*/result_batch_size,
        /*site=*/reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        /*kind=*/reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        /*coef=*/reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        /*configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
        /*psi=*/reinterpret_cast<const std::array<double, 2>*>(psi.data_ptr()),
        /*result_configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(sorted_result_configs.data_ptr()),
        /*result_psi=*/reinterpret_cast<std::array<double, 2>*>(sorted_result_psi.data_ptr())
    );
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto result_psi = torch::zeros_like(sorted_result_psi);
    result_psi.index_put_({result_sort_index}, sorted_result_psi);
    return result_psi;
}

// A record may have an odd byte stride. Read its packed weight only while its
// node is locked. The separate aligned root cutoff provides a cheap atomic
// rejection test for paths that cannot improve a full heap.
template<typename Row>
__device__ double path_weight(const Row& record) {
    double weight;
    for (std::size_t i = 0; i < sizeof(double); ++i) {
        reinterpret_cast<std::uint8_t*>(&weight)[i] = record[i];
    }
    return weight;
}

__device__ void lock_heap_node(int* address) {
    cuda::atomic_ref<int, cuda::thread_scope_device> lock(*address);
    unsigned int delay = 8;
    while (true) {
        int expected = 0;
        if (lock.compare_exchange_weak(expected, 1, cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed)) {
            return;
        }
        __nanosleep(delay);
        if (delay < 256) {
            delay *= 2;
        }
    }
}

__device__ void unlock_heap_node(int* address) {
    cuda::atomic_ref<int, cuda::thread_scope_device>(*address).store(0, cuda::std::memory_order_release);
}

template<typename Row>
__device__ void add_into_heap(Row* heap, int* mutex, double* root_weight, std::int64_t size, const Row& value, double weight) {
    cuda::atomic_ref<double, cuda::thread_scope_device> cutoff(*root_weight);
    if (!(weight > cutoff.load(cuda::std::memory_order_relaxed))) {
        return;
    }
    std::int64_t node = 0;
    lock_heap_node(&mutex[node]);
    if (!(weight > path_weight(heap[node]))) {
        unlock_heap_node(&mutex[node]);
        return;
    }
    while (node < size / 2) {
        const auto left = 2 * node + 1;
        const auto right = left + 1;
        lock_heap_node(&mutex[left]);
        auto child = left;
        if (right < size) {
            lock_heap_node(&mutex[right]);
            if (path_weight(heap[right]) < path_weight(heap[left])) {
                child = right;
            }
            unlock_heap_node(&mutex[child == left ? right : left]);
        }
        const double child_weight = path_weight(heap[child]);
        if (weight <= child_weight) {
            unlock_heap_node(&mutex[child]);
            break;
        }
        heap[node] = heap[child];
        if (node == 0) {
            cutoff.store(child_weight, cuda::std::memory_order_relaxed);
        }
        unlock_heap_node(&mutex[node]);
        node = child;
    }
    heap[node] = value;
    if (node == 0) {
        cutoff.store(weight, cuda::std::memory_order_relaxed);
    }
    unlock_heap_node(&mutex[node]);
}

// Extend the configuration subspace:
// 1. Generate new configurations.
// 2. Exclude existing configurations using binary search.
// 3. Compute weights and attempt insertion into the concurrent heap.
template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ void find_relative_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t exclude_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* exclude_configs, // exclude_size
    std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)>* heap,
    int* mutex,
    double* root_weight,
    std::int64_t heap_size
) {
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(
        /*current_configs=*/current_configs,
        /*term_index=*/term_index,
        /*site=*/site,
        /*kind=*/kind
    );

    if (!success) {
        return;
    }
    success = true;
    std::int64_t low = 0;
    std::int64_t high = exclude_size - 1;
    std::int64_t mid = 0;
    auto less = array_less<std::uint8_t, n_qubytes>();
    while (low <= high) {
        mid = (low + high) / 2;
        if (less(current_configs, exclude_configs[mid])) {
            high = mid - 1;
        } else if (less(exclude_configs[mid], current_configs)) {
            low = mid + 1;
        } else {
            success = false;
            break;
        }
    }
    if (!success) {
        return;
    }
    std::int8_t sign = parity ? -1 : +1;
    double real = sign * (coef[term_index][0] * psi[batch_index][0] - coef[term_index][1] * psi[batch_index][1]);
    double imag = sign * (coef[term_index][0] * psi[batch_index][1] + coef[term_index][1] * psi[batch_index][0]);
    // Rank individual Hamiltonian paths by their squared contribution.
    double weight = real * real + imag * imag;
    std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)> value;
    for (std::int64_t i = 0; i < sizeof(double) / sizeof(uint8_t); ++i) {
        value[i] = reinterpret_cast<const std::uint8_t*>(&weight)[i];
    }
    for (std::int64_t i = 0; i < n_qubytes; ++i) {
        value[i + sizeof(double) / sizeof(uint8_t)] = current_configs[i];
    }
    add_into_heap(heap, mutex, root_weight, heap_size, value, weight);
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void find_relative_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    std::int64_t exclude_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    const std::array<double, 2>* psi, // batch_size
    const std::array<std::uint8_t, n_qubytes>* exclude_configs, // exclude_size
    std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)>* heap,
    int* mutex,
    double* root_weight,
    std::int64_t heap_size
) {
    const std::int64_t term_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (term_index >= term_number) {
        return;
    }
    const auto batch_stride = static_cast<std::int64_t>(gridDim.y) * blockDim.y;
    for (std::int64_t batch_index = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         batch_index < batch_size; batch_index += batch_stride) {
        find_relative_kernel<max_op_number, n_qubytes, particle_cut>(
            /*term_index=*/term_index,
            /*batch_index=*/batch_index,
            /*term_number=*/term_number,
            /*batch_size=*/batch_size,
            /*exclude_size=*/exclude_size,
            /*site=*/site,
            /*kind=*/kind,
            /*coef=*/coef,
            /*configs=*/configs,
            /*psi=*/psi,
            /*exclude_configs=*/exclude_configs,
            /*heap=*/heap,
            /*mutex=*/mutex,
            /*root_weight=*/root_weight,
            /*heap_size=*/heap_size
        );
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut, bool return_pool = false>
auto find_relative_interface(
    const torch::Tensor& configs,
    const torch::Tensor& psi,
    const std::int64_t count_selected,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef,
    const torch::Tensor& exclude_configs
) -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t term_number = site.size(0);
    std::int64_t exclude_size = exclude_configs.size(0);

    at::cuda::CUDAGuard cuda_device_guard(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);

    TORCH_CHECK(configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(psi.device().type() == torch::kCUDA, "psi must be on CUDA.")
    TORCH_CHECK(psi.device().index() == device_id, "psi must be on the same device as others.");
    TORCH_CHECK(psi.is_contiguous(), "psi must be contiguous.")
    TORCH_CHECK(psi.dtype() == torch::kFloat64, "psi must be float64.")
    TORCH_CHECK(psi.dim() == 2, "psi must be 2D.")
    TORCH_CHECK(psi.size(0) == batch_size, "psi batch size must match the provided batch_size.");
    TORCH_CHECK(psi.size(1) == 2, "psi must contain 2 elements for each batch.");

    TORCH_CHECK(site.device().type() == torch::kCUDA, "site must be on CUDA.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCUDA, "kind must be on CUDA.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCUDA, "coef must be on CUDA.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    TORCH_CHECK(exclude_configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(exclude_configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(exclude_configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(exclude_configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(exclude_configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(exclude_configs.size(0) == exclude_size, "configs batch size must match the provided exclude_size.");
    TORCH_CHECK(exclude_configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    auto sorted_exclude_configs = exclude_configs.clone(torch::MemoryFormat::Contiguous);

    thrust::sort(
        policy,
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()),
        reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()) + exclude_size,
        array_less<std::uint8_t, n_qubytes>()
    );

    auto result_pool = torch::zeros(
        {count_selected, n_qubytes + sizeof(double) / sizeof(std::uint8_t)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device, device_id)
    );
    std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)>* heap =
        reinterpret_cast<std::array<std::uint8_t, n_qubytes + sizeof(double) / sizeof(std::uint8_t)>*>(result_pool.data_ptr());

    auto mutex = torch::zeros({count_selected}, configs.options().dtype(torch::kInt32));
    auto root_weight = torch::zeros({}, configs.options().dtype(torch::kFloat64));
    const auto [num_blocks, threads_per_block] = term_batch_launch(term_number, batch_size, device_id);
    find_relative_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks, threads_per_block, 0, stream>>>(
        /*term_number=*/term_number,
        /*batch_size=*/batch_size,
        /*exclude_size=*/exclude_size,
        /*site=*/reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        /*kind=*/reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        /*coef=*/reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        /*configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
        /*psi=*/reinterpret_cast<const std::array<double, 2>*>(psi.data_ptr()),
        /*exclude_configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()),
        /*heap=*/heap,
        /*mutex=*/mutex.data_ptr<int>(),
        /*root_weight=*/root_weight.data_ptr<double>(),
        /*heap_size=*/count_selected
    );
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Distributed selection needs the top individual paths (including repeated destinations).
    if constexpr (return_pool) {
        return result_pool;
    }

    // Here, the bytes before sizeof(double) / sizeof(std::uint8_t) in result_pool are weights, and the bytes after are configs
    // We need to remove items with weight 0, then sort and deduplicate the configs

    auto result_config =
        result_pool.index({torch::indexing::Slice(), torch::indexing::Slice(sizeof(double) / sizeof(std::uint8_t), torch::indexing::None)});
    auto result_weight =
        result_pool.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, sizeof(double) / sizeof(std::uint8_t))});
    auto nonzero = torch::any(result_weight != 0, /*dim=*/1);
    auto nonzero_result_config = result_config.index({nonzero});
    auto unique_nonzero_result_config =
        std::get<0>(torch::unique_dim(/*self=*/nonzero_result_config, /*dim=*/0, /*sorted=*/true, /*return_inverse=*/false, /*return_counts=*/false));
    return unique_nonzero_result_config;
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void enumerate_relative_tile(
    std::int64_t first_pair,
    std::int64_t count,
    std::int64_t batch_size,
    std::int64_t exclude_size,
    const std::array<std::int16_t, max_op_number>* site,
    const std::array<std::uint8_t, max_op_number>* kind,
    const std::array<double, 2>* coef,
    const std::array<std::uint8_t, n_qubytes>* configs,
    const std::array<double, 2>* psi,
    const std::array<std::uint8_t, n_qubytes>* exclude_configs,
    std::array<std::uint8_t, n_qubytes>* out_configs,
    std::array<double, 2>* out_psi,
    bool* valid
) {
    const std::int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) {
        return;
    }
    const auto term = (first_pair + row) / batch_size;
    const auto source = (first_pair + row) % batch_size;
    auto config = configs[source];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(config, term, site, kind);
    if (!success) {
        return;
    }

    std::int64_t low = 0, high = exclude_size - 1;
    auto less = array_less<std::uint8_t, n_qubytes>();
    while (low <= high) {
        auto mid = low + (high - low) / 2;
        if (less(config, exclude_configs[mid])) {
            high = mid - 1;
        } else if (less(exclude_configs[mid], config)) {
            low = mid + 1;
        } else {
            return;
        }
    }

    const double sign = parity ? -1 : 1;
    out_configs[row] = config;
    out_psi[row][0] = sign * (coef[term][0] * psi[source][0] - coef[term][1] * psi[source][1]);
    out_psi[row][1] = sign * (coef[term][0] * psi[source][1] + coef[term][1] * psi[source][0]);
    valid[row] = true;
}

using WeightedConfigs = std::pair<torch::Tensor, torch::Tensor>;

WeightedConfigs sum_equal_configs(const torch::Tensor& keys, const torch::Tensor& values) {
    auto [unique, inverse, counts] = torch::unique_dim(keys, 0, true, true, false);
    auto sums = torch::zeros({unique.size(0), 2}, values.options());
    sums.index_add_(0, inverse, values);
    return {unique, sums};
}

WeightedConfigs merge_weighted_configs(const WeightedConfigs& a, const WeightedConfigs& b) {
    return sum_equal_configs(torch::cat({a.first, b.first}), torch::cat({a.second, b.second}));
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto list_relative_interface(
    const torch::Tensor& configs,
    const torch::Tensor& psi,
    const torch::Tensor& site,
    const torch::Tensor& kind,
    const torch::Tensor& coef,
    const torch::Tensor& exclude_configs
) -> std::tuple<torch::Tensor, torch::Tensor> {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t term_number = site.size(0);
    std::int64_t exclude_size = exclude_configs.size(0);

    at::cuda::CUDAGuard cuda_device_guard(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);

    TORCH_CHECK(configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(psi.device().type() == torch::kCUDA, "psi must be on CUDA.")
    TORCH_CHECK(psi.device().index() == device_id, "psi must be on the same device as others.");
    TORCH_CHECK(psi.is_contiguous(), "psi must be contiguous.")
    TORCH_CHECK(psi.dtype() == torch::kFloat64, "psi must be float64.")
    TORCH_CHECK(psi.dim() == 2, "psi must be 2D.")
    TORCH_CHECK(psi.size(0) == batch_size, "psi batch size must match the provided batch_size.");
    TORCH_CHECK(psi.size(1) == 2, "psi must contain 2 elements for each batch.");

    TORCH_CHECK(site.device().type() == torch::kCUDA, "site must be on CUDA.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCUDA, "kind must be on CUDA.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCUDA, "coef must be on CUDA.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    TORCH_CHECK(exclude_configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(exclude_configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(exclude_configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(exclude_configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(exclude_configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(exclude_configs.size(0) == exclude_size, "configs batch size must match the provided exclude_size.");
    TORCH_CHECK(exclude_configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    if (!batch_size || !term_number) {
        return {torch::empty({0, n_qubytes}, configs.options()), torch::empty({0, 2}, psi.options())};
    }
    TORCH_CHECK(term_number <= std::numeric_limits<std::int64_t>::max() / batch_size, "Too many term/configuration pairs.");
    auto sorted_exclude_configs = exclude_configs.clone(torch::MemoryFormat::Contiguous);
    if (exclude_size > 1) {
        auto begin = reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr());
        thrust::sort(policy, begin, begin + exclude_size, array_less<std::uint8_t, n_qubytes>());
    }

    // Bound temporary candidate storage independently of the size of the Cartesian product.
    constexpr std::int64_t tile_size = 1 << 18;
    constexpr unsigned int threads = 256;
    const std::int64_t pairs = batch_size * term_number;
    std::vector<WeightedConfigs> levels;
    for (std::int64_t first = 0; first < pairs;) {
        const auto count = std::min(tile_size, pairs - first);
        auto keys = torch::empty({count, n_qubytes}, configs.options());
        auto values = torch::empty({count, 2}, psi.options());
        auto valid = torch::zeros({count}, configs.options().dtype(torch::kBool));
        const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
        enumerate_relative_tile<max_op_number, n_qubytes, particle_cut><<<blocks, threads, 0, stream>>>(
            first, count, batch_size, exclude_size,
            reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
            reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
            reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
            reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
            reinterpret_cast<const std::array<double, 2>*>(psi.data_ptr()),
            reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(sorted_exclude_configs.data_ptr()),
            reinterpret_cast<std::array<std::uint8_t, n_qubytes>*>(keys.data_ptr()),
            reinterpret_cast<std::array<double, 2>*>(values.data_ptr()),
            valid.data_ptr<bool>()
        );
        C10_CUDA_CHECK(cudaGetLastError());
        first += count;
        auto indices = torch::nonzero(valid).reshape({-1});
        if (indices.numel() == 0) {
            continue;
        }
        auto current = sum_equal_configs(keys.index_select(0, indices), values.index_select(0, indices));
        // Binary merging avoids re-sorting all previous tiles on every iteration.
        std::size_t level = 0;
        while (level < levels.size() && levels[level].first.defined()) {
            current = merge_weighted_configs(levels[level], current);
            levels[level] = {};
            ++level;
        }
        if (level == levels.size()) {
            levels.emplace_back();
        }
        levels[level] = std::move(current);
    }

    WeightedConfigs result;
    for (const auto& partial : levels) {
        if (partial.first.defined()) {
            result = result.first.defined() ? merge_weighted_configs(partial, result) : partial;
        }
    }
    if (!result.first.defined()) {
        return {torch::empty({0, n_qubytes}, configs.options()), torch::empty({0, 2}, psi.options())};
    }
    return {result.first, result.second};
}

// Compute the diagonal contribution of the Hamiltonian.
// Accumulate a term's complex coefficient when its action leaves the configuration unchanged.
template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__device__ void diagonal_term_kernel(
    std::int64_t term_index,
    std::int64_t batch_index,
    std::int64_t term_number,
    std::int64_t batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    std::array<double, 2>* result_psi
) {
    std::array<std::uint8_t, n_qubytes> current_configs = configs[batch_index];
    auto [success, parity] = hamiltonian_apply_kernel<max_op_number, n_qubytes, particle_cut>(
        /*current_configs=*/current_configs,
        /*term_index=*/term_index,
        /*site=*/site,
        /*kind=*/kind
    );

    if (!success) {
        return;
    }
    auto less = array_less<std::uint8_t, n_qubytes>();
    if (less(current_configs, configs[batch_index]) || less(configs[batch_index], current_configs)) {
        return; // The term does not apply to the current configuration
    }
    std::int8_t sign = parity ? -1 : +1;
    atomicAdd(&result_psi[batch_index][0], sign * coef[term_index][0]);
    atomicAdd(&result_psi[batch_index][1], sign * coef[term_index][1]);
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
__global__ void diagonal_term_kernel_interface(
    std::int64_t term_number,
    std::int64_t batch_size,
    const std::array<std::int16_t, max_op_number>* site, // term_number
    const std::array<std::uint8_t, max_op_number>* kind, // term_number
    const std::array<double, 2>* coef, // term_number
    const std::array<std::uint8_t, n_qubytes>* configs, // batch_size
    std::array<double, 2>* result_psi
) {
    const std::int64_t term_index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (term_index >= term_number) {
        return;
    }
    const auto batch_stride = static_cast<std::int64_t>(gridDim.y) * blockDim.y;
    for (std::int64_t batch_index = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         batch_index < batch_size; batch_index += batch_stride) {
        diagonal_term_kernel<max_op_number, n_qubytes, particle_cut>(
            /*term_index=*/term_index,
            /*batch_index=*/batch_index,
            /*term_number=*/term_number,
            /*batch_size=*/batch_size,
            /*site=*/site,
            /*kind=*/kind,
            /*coef=*/coef,
            /*configs=*/configs,
            /*result_psi=*/result_psi
        );
    }
}

template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>
auto diagonal_term_interface(const torch::Tensor& configs, const torch::Tensor& site, const torch::Tensor& kind, const torch::Tensor& coef)
    -> torch::Tensor {
    std::int64_t device_id = configs.device().index();
    std::int64_t batch_size = configs.size(0);
    std::int64_t term_number = site.size(0);

    at::cuda::CUDAGuard cuda_device_guard(device_id);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);
    auto policy = thrust::device.on(stream);

    TORCH_CHECK(configs.device().type() == torch::kCUDA, "configs must be on CUDA.")
    TORCH_CHECK(configs.device().index() == device_id, "configs must be on the same device as others.");
    TORCH_CHECK(configs.is_contiguous(), "configs must be contiguous.")
    TORCH_CHECK(configs.dtype() == torch::kUInt8, "configs must be uint8.")
    TORCH_CHECK(configs.dim() == 2, "configs must be 2D.")
    TORCH_CHECK(configs.size(0) == batch_size, "configs batch size must match the provided batch_size.");
    TORCH_CHECK(configs.size(1) == n_qubytes, "configs must have the same number of qubits as the provided n_qubytes.");

    TORCH_CHECK(site.device().type() == torch::kCUDA, "site must be on CUDA.")
    TORCH_CHECK(site.device().index() == device_id, "site must be on the same device as others.");
    TORCH_CHECK(site.is_contiguous(), "site must be contiguous.")
    TORCH_CHECK(site.dtype() == torch::kInt16, "site must be int16.")
    TORCH_CHECK(site.dim() == 2, "site must be 2D.")
    TORCH_CHECK(site.size(0) == term_number, "site size must match the provided term_number.");
    TORCH_CHECK(site.size(1) == max_op_number, "site must match the provided max_op_number.");

    TORCH_CHECK(kind.device().type() == torch::kCUDA, "kind must be on CUDA.")
    TORCH_CHECK(kind.device().index() == device_id, "kind must be on the same device as others.");
    TORCH_CHECK(kind.is_contiguous(), "kind must be contiguous.")
    TORCH_CHECK(kind.dtype() == torch::kUInt8, "kind must be uint8.")
    TORCH_CHECK(kind.dim() == 2, "kind must be 2D.")
    TORCH_CHECK(kind.size(0) == term_number, "kind size must match the provided term_number.");
    TORCH_CHECK(kind.size(1) == max_op_number, "kind must match the provided max_op_number.");

    TORCH_CHECK(coef.device().type() == torch::kCUDA, "coef must be on CUDA.")
    TORCH_CHECK(coef.device().index() == device_id, "coef must be on the same device as others.");
    TORCH_CHECK(coef.is_contiguous(), "coef must be contiguous.")
    TORCH_CHECK(coef.dtype() == torch::kFloat64, "coef must be float64.")
    TORCH_CHECK(coef.dim() == 2, "coef must be 2D.")
    TORCH_CHECK(coef.size(0) == term_number, "coef size must match the provided term_number.");
    TORCH_CHECK(coef.size(1) == 2, "coef must contain 2 elements for each term.");

    auto result_psi = torch::zeros({batch_size, 2}, torch::TensorOptions().dtype(torch::kFloat64).device(device, device_id));

    const auto [num_blocks, threads_per_block] = term_batch_launch(term_number, batch_size, device_id);
    diagonal_term_kernel_interface<max_op_number, n_qubytes, particle_cut><<<num_blocks, threads_per_block, 0, stream>>>(
        /*term_number=*/term_number,
        /*batch_size=*/batch_size,
        /*site=*/reinterpret_cast<const std::array<std::int16_t, max_op_number>*>(site.data_ptr()),
        /*kind=*/reinterpret_cast<const std::array<std::uint8_t, max_op_number>*>(kind.data_ptr()),
        /*coef=*/reinterpret_cast<const std::array<double, 2>*>(coef.data_ptr()),
        /*configs=*/reinterpret_cast<const std::array<std::uint8_t, n_qubytes>*>(configs.data_ptr()),
        /*result_psi=*/reinterpret_cast<std::array<double, 2>*>(result_psi.data_ptr())
    );
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    return result_psi;
}

#ifndef N_QUBYTES
#define N_QUBYTES 0
#endif
#ifndef PARTICLE_CUT
#define PARTICLE_CUT 0
#endif

#if N_QUBYTES != 0
#define HAAR_Q_LIBRARY_HELPER(x, y) haar_q_hamiltonian_##x##_##y
#define HAAR_Q_LIBRARY(x, y) HAAR_Q_LIBRARY_HELPER(x, y)
TORCH_LIBRARY_IMPL(HAAR_Q_LIBRARY(N_QUBYTES, PARTICLE_CUT), CUDA, m) {
    m.impl("apply_within", apply_within_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("find_relative", find_relative_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("relative_candidate_pool", find_relative_interface<4, N_QUBYTES, PARTICLE_CUT, true>);
    m.impl("list_relative", list_relative_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
    m.impl("diagonal_term", diagonal_term_interface</*max_op_number=*/4, /*n_qubytes=*/N_QUBYTES, /*particle_cut=*/PARTICLE_CUT>);
}
#undef HAAR_Q_LIBRARY
#undef HAAR_Q_LIBRARY_HELPER
#endif

} // namespace haar_q_hamiltonian_cuda
