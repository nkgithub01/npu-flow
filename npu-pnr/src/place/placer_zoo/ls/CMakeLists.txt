file(GLOB_RECURSE ls_placer_headers "*.hpp")
file(GLOB_RECURSE ls_placer_sources "*.cpp")

target_sources(npu_pnr
    PRIVATE ${ls_placer_sources}
    PUBLIC ${ls_placer_headers})
