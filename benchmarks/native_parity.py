"""Paired native apply-within timings for the 114-electron/146-site P-cluster.

Extracts production kernels verbatim, without requiring a CUDA-enabled PyTorch
wheel. Compilation, input reads, transfers and basis sorting are outside timing.
The two checkouts must expose the same native apply-within kernel interface.
"""

import argparse
import json
import pathlib
import re
import statistics
import struct
import subprocess
import tempfile
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline-checkout", type=pathlib.Path, required=True)
parser.add_argument("--current-checkout", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
parser.add_argument(
    "--cache-file", type=pathlib.Path, required=True, help="P-cluster packed v6 cache, from benchmarks/fcidump.py"
)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--compiler", help="g++ or nvcc executable; defaults to the selected device's compiler on PATH")
parser.add_argument("--cuda-arch", default="sm_75", help="CUDA device architecture (default: GTX 1650)")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--output-dir", type=pathlib.Path, help="Directory for generated sources, binaries and raw logs")
args = parser.parse_args()
if args.repeats < 1:
    parser.error("--repeats must be positive")
base = args.baseline_checkout.resolve()
current = args.current_checkout.resolve()
out = args.output_dir.resolve() if args.output_dir else pathlib.Path(tempfile.mkdtemp(prefix="haar-q-native-"))
out.mkdir(parents=True, exist_ok=True)
cache = args.cache_file.resolve()
with zipfile.ZipFile(cache) as z, cache.open("rb") as stream:
    entries = sorted((e for e in z.infolist() if "/data/" in e.filename), key=lambda e: e.filename)
    if len(entries) != 3:
        raise ValueError("Expected three packed tensor storages")
    offsets = []
    for e, stride in zip(entries, [8, 4, 16]):
        if e.file_size != 41528584 * stride or e.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Expected the real P-cluster v6 cache with 41,528,584 packed terms")
        stream.seek(e.header_offset + 26)
        name, extra = struct.unpack("<HH", stream.read(4))
        offsets.append(e.header_offset + 30 + name + extra)

header = r"""
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
"""
templ = "template<std::int64_t max_op_number, std::int64_t n_qubytes, std::int64_t particle_cut>"


def extracted(repo: pathlib.Path, gpu: bool) -> str:
    source = (
        repo / ("haar_q/hamiltonian/_hamiltonian_cuda.cu" if gpu else "haar_q/hamiltonian/_hamiltonian_cpu.cpp")
    ).read_text()
    start = source.index("template<typename T, std::int64_t size>")
    end = source.index(templ + "\nauto apply_within_interface")
    return source[start:end]


host = r"""
using Site = std::array<int16_t, 4>;
using Kind = std::array<uint8_t, 4>;
using Val = std::array<double, 2>;
using Cfg = std::array<uint8_t, 19>;
int main() {
  constexpr int chunk = 262144, total = 41528584;
  std::ifstream streams[3];
  for (int i = 0; i < 3; ++i) {
    streams[i].open(CACHE, std::ios::binary);
    streams[i].seekg(OFFSETS[i]);
  }
  Cfg cfg[8]{};
  Val psi[8];
  for (int p = 0; p < 114; ++p) cfg[0][p / 8] |= 1 << (p % 8);
  cfg[1] = cfg[0];
  cfg[1][14] = 6;
  std::mt19937 rng(1873);
  for (int b = 2; b < 8; ++b) {
    int occupied = 0;
    while (occupied < 114) {
      int p = rng() % 146;
      if (!(cfg[b][p / 8] & (1 << (p % 8)))) {
        cfg[b][p / 8] |= 1 << (p % 8);
        ++occupied;
      }
    }
  }
  std::sort(cfg, cfg + 8);
  for (int b = 0; b < 8; ++b) psi[b] = {1. + b * .125, b * .125};
  std::vector<Site> sb(chunk);
  std::vector<Kind> kb(chunk);
  std::vector<Val> cv(chunk);
  Val result[2][2][8]{};
  double elapsed[2][2]{};
  SETUP
  for (int start = 0; start < total; start += chunk) {
    int n = std::min(chunk, total - start);
    auto read = [n](auto& stream, auto& vec) {
      return bool(stream.read((char*)vec.data(), n * sizeof(vec[0])));
    };
    if (!(read(streams[0], sb) && read(streams[1], kb) && read(streams[2], cv))) return 3;
    TRANSFER
    for (int bi = 0; bi < 2; ++bi) {
      int batch = bi ? 8 : 2;
      for (int v = 0; v < 2; ++v) {
        int version = (v + (start / chunk)) % 2;
        RUN
      }
    }
  }
  COLLECT
  for (int bi = 0; bi < 2; ++bi) {
    int batch = bi ? 8 : 2;
    double err = 0;
    for (int b = 0; b < 8; ++b) {
      for (int c = 0; c < 2; ++c) {
        err = std::max(err, std::abs(result[0][bi][b][c] - result[1][bi][b][c]));
      }
    }
    std::cout << "batch=" << batch << " baseline_s=" << elapsed[0][bi]
              << " current_s=" << elapsed[1][bi]
              << " speedup=" << elapsed[0][bi] / elapsed[1][bi]
              << " max_error=" << err << std::endl;
    for (int b = 0; b < 8; ++b) {
      std::cout << "amplitude[" << bi << "][" << b << "]="
                << result[1][bi][b][0] << "," << result[1][bi][b][1] << std::endl;
    }
    if (err > 1e-8) return 4;
  }
}
"""
cpu_run = r"""
auto first = std::chrono::steady_clock::now();
if (version == 0) {
  old::apply_within_kernel_interface<4,19,1>(
      n, batch, 8, sb.data(), kb.data(), cv.data(), cfg, psi, cfg, result[0][bi]);
} else {
  now::apply_within_kernel_interface<4,19,1>(
      n, batch, 8, sb.data(), kb.data(), cv.data(), cfg, psi, cfg, result[1][bi]);
}
elapsed[version][bi] += std::chrono::duration<double>(std::chrono::steady_clock::now() - first).count();
"""
gpu_setup = r"""
cudaDeviceProp prop;
CHECK(cudaGetDeviceProperties(&prop, 0));
std::cout << "GPU=" << prop.name << std::endl;
Site *s;
Kind *k;
Val *c, *p, *res;
Cfg *configs;
CHECK(cudaMalloc(&s, chunk * sizeof(Site)));
CHECK(cudaMalloc(&k, chunk * sizeof(Kind)));
CHECK(cudaMalloc(&c, chunk * sizeof(Val)));
CHECK(cudaMalloc(&p, sizeof(psi)));
CHECK(cudaMalloc(&res, sizeof(result)));
CHECK(cudaMalloc(&configs, sizeof(cfg)));
CHECK(cudaMemcpy(p, psi, sizeof(psi), cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(configs, cfg, sizeof(cfg), cudaMemcpyHostToDevice));
CHECK(cudaMemset(res, 0, sizeof(result)));
cudaEvent_t first, last;
CHECK(cudaEventCreate(&first));
CHECK(cudaEventCreate(&last));
"""
gpu_transfer = r"""
CHECK(cudaMemcpy(s, sb.data(), n * sizeof(Site), cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(k, kb.data(), n * sizeof(Kind), cudaMemcpyHostToDevice));
CHECK(cudaMemcpy(c, cv.data(), n * sizeof(Val), cudaMemcpyHostToDevice));
"""
gpu_run = r"""
CHECK(cudaEventRecord(first));
dim3 threads(256 / batch, batch), blocks((n + threads.x - 1) / threads.x);
if (version == 0) {
  old::apply_within_kernel_interface<4,19,1><<<blocks,threads>>>(n, batch, 8, s, k, c, configs, p, configs, res + bi * 8);
} else {
  now::apply_within_kernel_interface<4,19,1><<<blocks,threads>>>(n, batch, 8, s, k, c, configs, p, configs, res + 16 + bi * 8);
}
CHECK(cudaEventRecord(last));
CHECK(cudaEventSynchronize(last));
float ms;
CHECK(cudaEventElapsedTime(&ms, first, last));
elapsed[version][bi] += ms * .001;
"""
for gpu in [args.device == "cuda"]:
    source = header
    if gpu:
        source += '#include <cuda_runtime.h>\n#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){std::cerr<<#x<<":"<<cudaGetErrorString(e)<<std::endl;return 2;}}while(0)\n'
    for label, repo in [("old", base), ("now", current)]:
        source += f"namespace {label} {{\n" + extracted(repo, gpu) + "\n}\n"
    source += (
        "\nconstexpr const char* CACHE="
        + json.dumps(str(cache))
        + ";\nconstexpr long OFFSETS[]={"
        + ",".join(map(str, offsets))
        + "};\n"
    )
    source += (
        host.replace("SETUP", gpu_setup if gpu else "")
        .replace("TRANSFER", gpu_transfer if gpu else "")
        .replace("RUN", gpu_run if gpu else cpu_run)
        .replace("COLLECT", "CHECK(cudaMemcpy(result,res,sizeof(result),cudaMemcpyDeviceToHost));" if gpu else "")
    )
    file = out / ("bench.cu" if gpu else "bench.cpp")
    file.write_text(source)
    if gpu:
        cmd = [
            args.compiler or "nvcc",
            "--std=c++20",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-O3",
            "--use_fast_math",
            f"-arch={args.cuda_arch}",
            str(file),
            "-o",
            str(out / "gpu"),
        ]
    else:
        cmd = [
            args.compiler or "g++",
            "-std=c++20",
            "-O3",
            "-ffast-math",
            "-march=native",
            str(file),
            "-o",
            str(out / "cpu"),
        ]
    subprocess.run(cmd, check=True)
    runs = []
    pattern = r"batch=(\d+) baseline_s=([\d.e+-]+) current_s=([\d.e+-]+) speedup=([\d.e+-]+) max_error=([\d.e+-]+)"
    for repeat in range(args.repeats):
        result = subprocess.run([str(out / ("gpu" if gpu else "cpu"))], check=True, capture_output=True, text=True)
        (out / f"run-{repeat + 1}.txt").write_text(result.stdout)
        samples = [
            dict(batch=int(b), baseline_s=float(old), current_s=float(new), speedup=float(speed), max_error=float(err))
            for b, old, new, speed, err in re.findall(pattern, result.stdout)
        ]
        if len(samples) != 2:
            raise RuntimeError(f"Expected timings for both basis sizes: {result.stdout}")
        runs.append(samples)
    summary = []
    for index in range(2):
        baseline = statistics.median(run[index]["baseline_s"] for run in runs)
        optimized = statistics.median(run[index]["current_s"] for run in runs)
        summary.append(
            dict(
                batch=runs[0][index]["batch"],
                baseline_median_s=baseline,
                current_median_s=optimized,
                speedup=baseline / optimized,
                max_error=max(run[index]["max_error"] for run in runs),
            )
        )
    print(
        json.dumps(
            dict(
                device=args.device,
                terms=41528584,
                spin_orbitals=146,
                electrons=114,
                result_basis_size=8,
                kernel_only=True,
                alternating_order_per_tile=True,
                tile_terms=262144,
                compile_command=cmd,
                baseline_checkout=str(base),
                current_checkout=str(current),
                cache_file=str(cache),
                output_dir=str(out),
                runs=runs,
                summary=summary,
            ),
            indent=2,
        )
    )
