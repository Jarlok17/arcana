add_library(ArcanaTensor INTERFACE)

# Alias
add_library(Arcana::Tensor ALIAS ArcanaTensor)

# Include directories
target_include_directories(ArcanaTensor
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# C++23 standart
target_compile_features(ArcanaTensor INTERFACE cxx_std_23)

# OpenMP for SIMD
find_package(OpenMP REQUIRED)
target_link_libraries(ArcanaTensor INTERFACE OpenMP::OpenMP_CXX)

# Compiler flags for SIMD
if(NOT MSVC)
	target_compile_options(ArcanaTensor INTERFACE
        -O3
        -march=native
    )
else()
	target_compile_options(ArcanaTensor INTERFACE
        /O2
        /arch:AVX2
    )
endif()
