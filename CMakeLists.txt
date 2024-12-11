cmake_minimum_required(VERSION 3.14)

# 项目信息
project(ckks LANGUAGES CUDA CXX)


# 设置 CUDA 版本
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w") # 全局设置



# 添加共享库
add_library(lib SHARED
    helper.cuh
    encoder.cuh
    parameter.cuh
    uint128.cuh
    encryptor.cuh
    polycalc.cuh
    random.cuh
    evaluator.cuh
    rns.cuh
    ntt_60bit.cuh
    rottable.cuh
    library.cuh
    library.cu
    )

set_target_properties(lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
    POSITION_INDEPENDENT_CODE ON  # 确保生成的代码是位置无关的
)
# 添加可执行文件
add_executable(ckkstest ckkstest.cu)
set_target_properties(ckkstest PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)

set(Torch_DIR ~/work/libtorch/share/cmake/Torch/)
find_package(Torch REQUIRED)


# 链接库
target_link_libraries(ckkstest lib)

target_link_libraries(ckkstest ${TORCH_LIBRARIES})

# 设置库和可执行文件的输出目录
set_target_properties(lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)


set_target_properties(ckkstest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(ckkstest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 显示所有符号
set_target_properties(lib PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

set_target_properties(ckkstest PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)