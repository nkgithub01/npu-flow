#include "route/router_zoo/milp/constraint.hpp"

#include <format>

#include "base/abstraction.hpp"

namespace route::router_zoo::milp {

void addCoreMappingConstraints(
    math_opt::Model &model, const LogicalPhysicalCoreToVarLookup &d,
    const base::LogicalCoreSet &l_cores, const base::PhysicalCoreSet &p_cores,
    const base::LogicalToPhysicalCoreAvailablityTable &l_to_p_core_avail_table,
    const base::LogicalCoreCompatibilitySet &l_core_compat_set,
    const std::vector<base::LogicalToPhysicalCoreMapping>
        &l_to_p_core_mapping_blacklist) {

  using base::LogicalCore;
  using base::PhysicalCore;

  // 1. Each logical core is mapped to exactly one physical core
  for (const LogicalCore &l_core : l_cores) {
    std::vector<math_opt::Variable> vars; // guaranteed non-empty
    for (const PhysicalCore &p_core : l_to_p_core_avail_table.at(l_core)) {
      vars.push_back(d.at({l_core, p_core}));
    }
    model.AddLinearConstraint(
        // Sum_{p in Avail(l)} d_{l,p} == 1
        math_opt::Sum(vars) == 1,
        std::format("Logical core {} mapped to exactly one physical core",
                    l_core.getName()));

    for (const PhysicalCore &p_core : p_cores) {
      if (!l_to_p_core_avail_table.at(l_core).contains(p_core)) {
        // Logical core l cannot be mapped to physical core p
        model.AddLinearConstraint(
            // d_{l,p} == 0
            d.at({l_core, p_core}) == 0,
            std::format("Logical core {} cannot be mapped to physical core {}",
                        l_core.getName(), p_core.getName()));
      }
    }
  }

  // 2. Logical core compatibility constraints
  for (const PhysicalCore &p_core : p_cores) {
    std::vector<math_opt::Variable> aux_vars;

    for (size_t subset_id = 0;
         const auto &subset : l_core_compat_set.getSubsets()) {
      aux_vars.push_back(model.AddBinaryVariable(
          std::format("l_compat_aux_{}_{}", subset_id, p_core.getName())));

      std::vector<math_opt::Variable> vars; // guaranteed non-empty
      for (const LogicalCore &l_core : subset) {
        vars.push_back(d.at({l_core, p_core}));
      }

      model.AddLinearConstraint(
          // Sum_{l in subset} d_{l,p} <= |subset| * aux_var
          math_opt::Sum(vars) <= int(subset.size()) * aux_vars.back(),
          std::format("Logical core compatibility constraint on physical core "
                      "{} for subset {}",
                      p_core.getName(), subset_id));

      subset_id++;
    }

    if (!aux_vars.empty()) {
      model.AddLinearConstraint(
          // Sum_{subsets} aux_var <= 1
          math_opt::Sum(aux_vars) <= 1,
          std::format(
              "At most one logical core compatibility subset on physical "
              "core {}",
              p_core.getName()));
    }
  }

  // 3. Logical to physical core mapping blacklist constraints
  for (int i = 0; const auto &core_mapping : l_to_p_core_mapping_blacklist) {
    std::vector<math_opt::Variable> vars;
    for (const auto &[l_core, p_core] : core_mapping.data()) {
      // The mapped p_core may not in the availablity table which determines d
      if (d.contains({l_core, p_core})) {
        vars.push_back(d.at({l_core, p_core}));
      } else {
        // If any (l_core, p_core) pair in the blacklist mapping is invalid or
        // not available according to the availablity table, then the whole
        // mapping will be unavailable and we can skip adding the constraint
        vars.clear();
        break;
      }
    }

    if (!vars.empty()) {
      model.AddLinearConstraint(
          // Sum_{(l,p) in mapping} d_{l,p} <= |mapping| - 1
          math_opt::Sum(vars) <= int(vars.size()) - 1,
          std::format(
              "Logical to physical core mapping blacklist constraint [{}]",
              i++));
    }
  }
}

} // namespace route::router_zoo::milp
