file(GLOB_RECURSE sa_placer_headers "*.hpp")
file(GLOB_RECURSE sa_placer_sources "*.cpp")

target_sources(npu_pnr
    PRIVATE ${sa_placer_sources}
    PUBLIC ${sa_placer_headers})
