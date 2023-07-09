#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

// Adaptive median filter
uchar adaptiveMedianFilter(cv::Mat& img, int row, int col, int kernelSize, int maxSize)
{
    std::vector<uchar> pixels;
    for (int y = -kernelSize / 2; y <= kernelSize / 2; y++)
    {
        for (int x = -kernelSize / 2; x <= kernelSize / 2; x++)
        {
            pixels.push_back(img.at<uchar>(row + y, col + x)); // Collect pixel values within the window
        }
    }

    std::sort(pixels.begin(), pixels.end()); // Sort the pixel values within the window

    auto min = pixels[0]; // min
    auto max = pixels[kernelSize * kernelSize - 1]; // max
    auto med = pixels[kernelSize * kernelSize / 2]; // median
    auto zxy = img.at<uchar>(row, col); // Current pixel value

    if (med > min && med < max)
    {
        // If the median is not noise, determine the return center point value or median based on the current pixel
        if (zxy > min && zxy < max)
            return zxy;
        else
            return med;
    }
    else
    {
        kernelSize += 2; // Increase window size
        if (kernelSize <= maxSize)
            return adaptiveMedianFilter(img, row, col, kernelSize, maxSize); // Increase the window size and continue with process A.
        else
            return med;
    }
}

// Adaptive mean filter
void adaptiveMeanFilter(const cv::Mat& src, cv::Mat& dst, int minSize = 3, int maxSize = 7)
{
    cv::copyMakeBorder(src, dst, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BORDER_REFLECT); // Fill the input image with boundaries
    int rows = dst.rows;
    int cols = dst.cols;

    // Start measuring time
    double start = omp_get_wtime();

#pragma omp parallel for num_threads(8)
    for (int j = maxSize / 2; j < rows - maxSize / 2; ++j)
    {
        for (int i = maxSize / 2; i < cols * dst.channels() - maxSize / 2; ++i)
        {
            dst.at<uchar>(j, i) = adaptiveMedianFilter(dst, j, i, minSize, maxSize); // Application of adaptive Median filter
        }
    }

    // Stop measuring time
    double end = omp_get_wtime();

    // Calculate the elapsed time
    double elapsedSeconds = end - start;

    std::cout << "Elapsed time: " << elapsedSeconds << " seconds" << std::endl;
}

int main()
{
    cv::Mat src = cv::imread("hehua.jpg"); // Read Image
    if (src.empty())
    {
        std::cout << "Failed to open image" << std::endl;
        return -1;
    }

    cv::Mat dst;

    adaptiveMeanFilter(src, dst); // Applying Adaptive Mean Filter

    cv::imshow("src", src); // Show original image
    cv::imshow("dst", dst); // Display filtering results
    cv::waitKey(0);

    return 0;
}

