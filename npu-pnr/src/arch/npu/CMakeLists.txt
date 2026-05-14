find_package(ortools CONFIG REQUIRED)

file(GLOB_RECURSE arch_npu_srcs "*.cpp")

target_sources(npu_pnr
    PRIVATE ${arch_npu_srcs}
    PUBLIC npu.hpp tile.hpp)

target_include_directories(npu_pnr PRIVATE ${PROJECT_SOURCE_DIR}/external_libs)

target_link_libraries(npu_pnr PRIVATE ortools::ortools)
