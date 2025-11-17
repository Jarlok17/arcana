cmake_minimum_required(VERSION 3.20)
project(Arcana VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(ARACANA_BUILD_VELVETROOM "Build tests" ON)
option(ARACANA_BUILD_SANDBOX "Build sandbox application" ON)
option(ARACANA_BUILD_DOCS "Build documentation" OFF)

if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic -Werror)
endif()

add_subdirectory(arcana)

if(ARACANA_BUILD_SANDBOX)
    add_subdirectory(sandbox)
endif()

if(ARACANA_BUILD_VELVETROOM)
    enable_testing()
    add_subdirectory(tests)
endif()
