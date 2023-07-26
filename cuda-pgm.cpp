#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cuda_runtime.h>

typedef unsigned char uchar;

#define CHECK_CUDA_ERROR() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

class SimpleImage {
public:
    int width, height;
    uchar* data; // Use pointer for CUDA memory allocation

    SimpleImage() : width(0), height(0), data(nullptr) {}

    ~SimpleImage() {
        if (data) {
            delete[] data;
        }
    }

    __device__ uchar& at(int row, int col) {
        return data[row * width + col];
    }
};

SimpleImage readPGM(const std::string& path) {
    SimpleImage img;
    std::ifstream file(path, std::ios::binary);
    std::string line;
    std::getline(file, line);  // Read the magic number
    if (line != "P5") {
        std::cerr << "Not a valid PGM file!" << std::endl;
        return img;
    }
    std::getline(file, line);  // Possibly read the comment line
    while (line[0] == '#') {
        std::getline(file, line);
    }
    std::stringstream ss(line);
    ss >> img.width >> img.height;
    int max_val;
    file >> max_val;
    file.get();  // Consume the newline
    img.data = new uchar[img.width * img.height];
    file.read(reinterpret_cast<char*>(img.data), img.width * img.height);
    return img;
}

void writePGM(const SimpleImage& img, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    file << "P5\n";
    file << "# Created by our program\n";
    file << img.width << " " << img.height << "\n";
    file << "255\n";
    file.write(reinterpret_cast<const char*>(img.data), img.width * img.height);
}

__device__ uchar adaptiveMedianFilter(SimpleImage& img, int row, int col, int minSize, int maxSize) {
    int kernelSize = minSize;
    uchar pixels[49]; // Assuming maxSize is 7, 7*7 = 49
    while (kernelSize <= maxSize) {
        int count = 0;
        for (int y = -kernelSize / 2; y <= kernelSize / 2; y++) {
            for (int x = -kernelSize / 2; x <= kernelSize / 2; x++) {
                pixels[count++] = img.at(row + y, col + x); // Collect pixel values within the window
            }
        }

        // Simple insertion sort
        for (int i = 1; i < count; i++) {
            uchar key = pixels[i];
            int j = i - 1;
            while (j >= 0 && pixels[j] > key) {
                pixels[j + 1] = pixels[j];
                j = j - 1;
            }
            pixels[j + 1] = key;
        }

        auto min = pixels[0]; // min
        auto max = pixels[count - 1]; // max
        auto med = pixels[count / 2]; // median
        auto zxy = img.at(row, col); // Current pixel value

        if (med > min && med < max) {
            if (zxy > min && zxy < max)
                return zxy;
            else
                return med;
        }
        else {
            kernelSize += 2; // Increase window size
        }
    }
    return img.at(row, col); // Default return if all else fails
}


__global__ void adaptiveMeanFilterKernel(SimpleImage src, SimpleImage dst, int minSize, int maxSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= maxSize / 2 && i < src.width - maxSize / 2 && j >= maxSize / 2 && j < src.height - maxSize / 2) {
        dst.at(j, i) = adaptiveMedianFilter(dst, j, i, minSize, maxSize);
    }
}

void adaptiveMeanFilter(const SimpleImage& src, SimpleImage& dst, int minSize = 3, int maxSize = 7) {
    dim3 block(16, 16);
    dim3 grid((src.width + block.x - 1) / block.x, (src.height + block.y - 1) / block.y);

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    adaptiveMeanFilterKernel << <grid, block >> > (src, dst, minSize, maxSize);
    CHECK_CUDA_ERROR();  // Check for errors after kernel launch

    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR();  // Check for errors after device synchronization

    // Stop measuring time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
}

int main() {
    SimpleImage src = readPGM("hehua-pgm.pgm");
    if (!src.data) {
        std::cout << "Failed to open image" << std::endl;
        return -1;
    }

    SimpleImage dst;
    dst.width = src.width;
    dst.height = src.height;
    cudaMalloc(&dst.data, dst.width * dst.height * sizeof(uchar));
    CHECK_CUDA_ERROR();  // Check for errors after memory allocation

    adaptiveMeanFilter(src, dst);

    writePGM(dst, "output.pgm");

    std::cout << "Processed image saved to output.pgm" << std::endl;

    cudaFree(dst.data);
    CHECK_CUDA_ERROR();  // Check for errors after freeing memory

    delete[] src.data;
    return 0;
}