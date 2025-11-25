include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.11.0
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(Catch2)

add_executable(arcana_tests
    tensor_test.cpp
    static_tensor_test.cpp
)

target_link_libraries(arcana_tests PRIVATE
    Arcana::Tensor
    Catch2::Catch2WithMain
)

target_include_directories(arcana_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/nebula/tensor/include
)

find_package(OpenMP REQUIRED)
target_link_libraries(arcana_tests PRIVATE OpenMP::OpenMP_CXX)

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(arcana_tests PRIVATE
        -Wall -Wextra
        -Wno-array-bounds
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(arcana_tests PRIVATE
        -Wall -Wextra -Wpedantic
    )
elseif(MSVC)
    target_compile_options(arcana_tests PRIVATE
        /W4
    )
endif()

include(CTest)
include(Catch)
catch_discover_tests(arcana_tests)