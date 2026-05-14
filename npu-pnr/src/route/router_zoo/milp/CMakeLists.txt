find_package(ortools CONFIG REQUIRED)

file(GLOB_RECURSE milp_router_headers "*.hpp")
file(GLOB_RECURSE milp_router_sources "*.cpp")

target_sources(npu_pnr
    PRIVATE ${milp_router_sources}
    PUBLIC ${milp_router_headers})

target_link_libraries(npu_pnr PRIVATE ortools::ortools)
