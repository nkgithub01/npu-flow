file(GLOB_RECURSE base_header "*.hpp")

target_sources(npu_pnr PUBLIC ${base_header})

target_include_directories(npu_pnr PUBLIC ${PROJECT_SOURCE_DIR}/external_libs)
