#include "tensor.h"
#include <iostream>
#include <chrono>

int main() {
    // Matrix dimensions: A(2x3), B(3x2), C(2x2)
    Tensor A(2, 3);
    Tensor B(3, 2);
    Tensor C(2, 2);
    
    // Initialize with sample data
    A.initialize({1, 2, 3, 
                 4, 5, 6});
    B.initialize({7, 8, 
                 9, 10,
                 11, 12});
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Perform matrix multiplication
        Tensor::multiply(A, B, C);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        
        std::cout << "Matrix A:\n";
        A.print();
        
        std::cout << "\nMatrix B:\n";
        B.print();
        
        std::cout << "\nResult C (A * B):\n";
        C.print();
        
        #ifdef USE_CUDA
        std::cout << "\nComputation time (with CUDA): ";
        #else
        std::cout << "\nComputation time (CPU only): ";
        #endif
        std::cout << duration.count() << " seconds\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}