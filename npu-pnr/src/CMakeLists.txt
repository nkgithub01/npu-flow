add_library(npu_pnr)

target_include_directories(npu_pnr PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_compile_features(npu_pnr PUBLIC cxx_std_20)

add_subdirectory(arch)
add_subdirectory(base)
add_subdirectory(engine)
add_subdirectory(place)
add_subdirectory(route)
add_subdirectory(utils)
