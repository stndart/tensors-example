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
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Fetch GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find all source files
file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
# Remove main.cpp from the list
list(FILTER CPP_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/*.cpp")
if (ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "src/*.cu")
endif()

# Create executable
add_executable(TensorMultiply src/main.cpp ${CPP_SOURCES} ${CUDA_SOURCES})

# Create test executable
add_executable(TensorMultiplyTests ${CPP_SOURCES} ${TEST_SOURCES} ${CUDA_SOURCES})

# Link GTest to test executable
target_link_libraries(TensorMultiplyTests PRIVATE gtest_main gtest)

# Set CUDA architecture for both targets
if(ENABLE_CUDA)
    set_target_properties(TensorMultiply TensorMultiplyTests PROPERTIES
        CUDA_ARCHITECTURES "86"
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Handle platform-specific linking
if(UNIX AND NOT APPLE)
    target_link_libraries(TensorMultiply pthread)
    target_link_libraries(TensorMultiplyTests pthread)
endif()

# Optional: Add alias for convenience
add_custom_target(runtests COMMAND TensorMultiplyTests DEPENDS TensorMultiplyTests)