# Grounding DINO runtime for the Modalix DevKit (build on the board:
#   cmake -S runtime -B runtime/build && cmake --build runtime/build -j8).
# SimaLMM (the LLiMa MLA runtime) is a DevKit package: sima-lmm-dev.
cmake_minimum_required(VERSION 3.16)
project(dino_runtime CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(SimaLMM REQUIRED)
find_package(OpenMP REQUIRED)
# SimaLMM's headers include <json/json.h> (jsoncpp) from /usr/include/jsoncpp.
find_path(JSONCPP_INCLUDE_DIR json/json.h PATHS /usr/include/jsoncpp /usr/local/include/jsoncpp)
set(DINO_FLAGS -O3 -march=armv8.2-a+fp16 -Wall -Wextra)

add_executable(elf_run tests/elf_run.cpp)
target_include_directories(elf_run PRIVATE include ${JSONCPP_INCLUDE_DIR})
target_link_libraries(elf_run PRIVATE SimaLMM::sima_lmm_runtime)
target_compile_options(elf_run PRIVATE ${DINO_FLAGS})

add_executable(dino_rt src/dino_rt.cpp)
target_include_directories(dino_rt PRIVATE include ${JSONCPP_INCLUDE_DIR})
target_link_libraries(dino_rt PRIVATE SimaLMM::sima_lmm_runtime OpenMP::OpenMP_CXX)
target_compile_options(dino_rt PRIVATE ${DINO_FLAGS})
install(TARGETS dino_rt RUNTIME DESTINATION bin)

add_executable(msda_bench tests/msda_bench.cpp)
target_include_directories(msda_bench PRIVATE include)
target_link_libraries(msda_bench PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(msda_bench PRIVATE ${DINO_FLAGS})
