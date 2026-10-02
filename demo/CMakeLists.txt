cmake_minimum_required(VERSION 3.15)
project(LRSchedulerDemo)

# 手动查找 gflags 头文件和库文件
find_path(GFLAGS_INCLUDE_DIR NAMES gflags/gflags.h)
find_library(GFLAGS_LIBRARY NAMES gflags)

if(NOT GFLAGS_INCLUDE_DIR OR NOT GFLAGS_LIBRARY)
    message(FATAL_ERROR "Could not find gflags. Please install libgflags-dev or set CMAKE_PREFIX_PATH.")
endif()

# 添加可执行文件
add_executable(lr_scheduler_demo examples/main.cc)

# 包含头文件目录
target_include_directories(lr_scheduler_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GFLAGS_INCLUDE_DIR}
)

# 链接 gflags 库
target_link_libraries(lr_scheduler_demo ${GFLAGS_LIBRARY})

# 设置 C++ 标准
set_target_properties(lr_scheduler_demo PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
)