file(GLOB_RECURSE milp_placer_headers "*.hpp")
file(GLOB_RECURSE milp_placer_sources "*.cpp")

target_sources(npu_pnr
    PRIVATE ${milp_placer_sources}
    PUBLIC ${milp_placer_headers})
