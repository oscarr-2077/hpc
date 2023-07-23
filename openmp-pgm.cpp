#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>

typedef unsigned char uchar;

class SimpleImage {
public:
    int width, height;
    std::vector<uchar> data;

    uchar& at(int row, int col) {
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
    img.data.resize(img.width * img.height);
    file.read(reinterpret_cast<char*>(img.data.data()), img.width * img.height);
    return img;
}

void writePGM(const SimpleImage& img, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    file << "P5\n";
    file << "# Created by our program\n";
    file << img.width << " " << img.height << "\n";
    file << "255\n";
    file.write(reinterpret_cast<const char*>(img.data.data()), img.width * img.height);
}

uchar adaptiveMedianFilter(SimpleImage& img, int row, int col, int kernelSize, int maxSize) {
    std::vector<uchar> pixels;
    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++) {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++) {
            pixels.push_back(img.at(row + y, col + x)); // Collect pixel values within the window
        }
    }

    std::sort(pixels.begin(), pixels.end()); // Sort the pixel values within the window

    auto min = pixels[0]; // min
    auto max = pixels[kernelSize * kernelSize - 1]; // max
    auto med = pixels[kernelSize * kernelSize / 2]; // median
    auto zxy = img.at(row, col); // Current pixel value

    if (med > min && med < max) {
        // If the median is not noise, determine the return center point value or median based on the current pixel
        if (zxy > min && zxy < max)
            return zxy;
        else
            return med;
    }
    else {
        kernelSize += 2; // Increase window size
        if (kernelSize <= maxSize)
            return adaptiveMedianFilter(img, row, col, kernelSize, maxSize); // Increase the window size and continue with process A.
        else
            return med;
    }
}

void adaptiveMeanFilter(const SimpleImage& src, SimpleImage& dst, int minSize = 3, int maxSize = 7) {
    // Assuming the dst image is already initialized with the same size as src
    int rows = dst.height;
    int cols = dst.width;

    // Start measuring time
    double start = omp_get_wtime();

#pragma omp parallel for num_threads(8)
    for (int j = maxSize / 2; j < rows - maxSize / 2; ++j) {
        for (int i = maxSize / 2; i < cols - maxSize / 2; ++i) {
            dst.at(j, i) = adaptiveMedianFilter(dst, j, i, minSize, maxSize); // Application of adaptive Median filter
        }
    }

    std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;

    // Stop measuring time
    double end = omp_get_wtime();

    // Calculate the elapsed time
    double elapsedSeconds = end - start;

    std::cout << "Elapsed time: " << elapsedSeconds << " seconds" << std::endl;
}


int main() {
    SimpleImage src = readPGM("hehua-pgm.pgm");
    if (src.data.empty()) {
        std::cout << "Failed to open image" << std::endl;
        return -1;
    }

    SimpleImage dst;
    dst.width = src.width;
    dst.height = src.height;
    dst.data.resize(dst.width * dst.height);

    adaptiveMeanFilter(src, dst);

    writePGM(dst, "output.pgm");

    std::cout << "Processed image saved to output.pgm" << std::endl;

    return 0;
}
