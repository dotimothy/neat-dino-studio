// Run one bare MLA ELF through SimaLMM's MLAModelWithBuffer and time it.
//   elf_run <model.elf> <in.f32> <in_elems> <out.f32> <out_elems> [iters]
// Input/output files are raw float32 in the ELF's HWC element order; the tool
// converts to/from bf16 in the device buffers.
#include <sima_lmm/mla_buffer.hpp>
#include <sima_lmm/mla_model.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include "dino/bf16.hpp"

namespace llima = simaai::llima;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: elf_run model.elf in.f32 in_elems out.f32 out_elems [iters]\n");
    return 2;
  }
  const size_t nin = std::stoul(argv[3]), nout = std::stoul(argv[5]);
  const int iters = argc > 6 ? std::stoi(argv[6]) : 20;
  std::vector<float> in(nin), out(nout);
  std::ifstream(argv[2], std::ios::binary).read((char*)in.data(), nin * 4);
  llima::connect_mla_rt({});
  {
    llima::MLABuffer ib("ifm", {1, nin}, "bfloat16", false), ob("ofm", {1, nout}, "bfloat16", false);
    ib.allocate();
    ob.allocate();
    ob.clear();
    llima::MLAModelWithBuffer m(argv[1], {llima::MLABufferSlice(&ib)}, {llima::MLABufferSlice(&ob)});
    auto t0 = Clock::now();
    m.load();
    std::printf("load %.1f ms\n", std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    dino::to_bf16_span(in.data(), (uint16_t*)ib.get_virtual_addr(), nin);
    ib.flush_cache();
    m.run();
    std::vector<double> ms;
    for (int i = 0; i < iters; ++i) {
      t0 = Clock::now();
      m.run();
      ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    ob.invalidate_cache();
    dino::from_bf16_span((const uint16_t*)ob.get_virtual_addr(), out.data(), nout);
    std::sort(ms.begin(), ms.end());
    std::printf("run median %.3f ms min %.3f ms (%d iters)\n", ms[ms.size() / 2], ms[0], iters);
    std::ofstream(argv[4], std::ios::binary).write((const char*)out.data(), nout * 4);
    m.free();
    ib.free();
    ob.free();
  }
  llima::disconnect_mla_rt();
  return 0;
}
