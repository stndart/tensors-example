#include <gtest/gtest.h>

int main(int argc, char **argv) {
#ifdef USE_CUDA
    std::cout << "Running tests with CUDA support" << std::endl;
#else
    std::cout << "Running tests without CUDA support" << std::endl;
#endif

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}