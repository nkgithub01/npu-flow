#pragma once

#include "arch/arch.hpp"

#include "arch/npu/npu.hpp"

namespace arch {

inline Arch createArch(const base::Config &cfg) {
  std::string type = cfg.get<std::string>("type");
  if (type != "npu") {
    throw std::runtime_error("Arch type " + type + " is not implemented.");
  }
  return Arch::create<arch::npu::NPU>(cfg);
}

} // namespace arch
