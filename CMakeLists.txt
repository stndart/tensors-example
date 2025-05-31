cmake_minimum_required(VERSION 3.18)

# Add CUDA enable/disable option
option(ENABLE_CUDA "Enable CUDA support" ON)

if(ENABLE_CUDA)
    project(TensorMultiply LANGUAGES CXX CUDA)
    message(STATUS "CUDA support enabled")
    add_compile_definitions(USE_CUDA)
else()
project(TensorMultiply LANGUAGES CXX)
    message(STATUS "CUDA support disabled")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find all source files automatically
file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
if (ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "src/*.cu")
else()
endif()

# Create executable
add_executable(TensorMultiply ${CPP_SOURCES} ${CUDA_SOURCES})

# Set CUDA architecture (adjust based on your GPU)
set_target_properties(TensorMultiply PROPERTIES
    CUDA_ARCHITECTURES "75"  # Turing (RTX 20xx), adjust as needed
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable CUDA support in code
target_compile_definitions(TensorMultiply PRIVATE)

# Handle platform-specific linking
if(UNIX AND NOT APPLE)
    target_link_libraries(TensorMultiply pthread)
endif()