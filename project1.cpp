#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>

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

    for (int j = maxSize / 2; j < rows - maxSize / 2; ++j)
    {
        for (int i = maxSize / 2; i < cols * dst.channels() - maxSize / 2; ++i)
        {
            dst.at<uchar>(j, i) = adaptiveMedianFilter(dst, j, i, minSize, maxSize); // Application of adaptive Median filter
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2)
    {
        std::cerr << "At least 2 processes are required." << std::endl;
        MPI_Finalize();
        return -1;
    }

    cv::Mat src;
    if (rank == 0)
    {
        src = cv::imread("hehua.jpg"); // Read Image
        if (src.empty())
        {
            std::cerr << "Failed to open image" << std::endl;
            MPI_Finalize();
            return -1;
        }
    }

    // Broadcast image dimensions to all processes
    int rows, cols;
    if (rank == 0)
    {
        rows = src.rows;
        cols = src.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute workload for each process
    int startRow = rank * (rows / (size - 1));
    int endRow = (rank == size - 1) ? rows : ((rank + 1) * (rows / (size - 1)));

    cv::Mat localSrc(endRow - startRow, cols, CV_8UC3);
    cv::Mat localDst;

    // Scatter image data to all processes
    MPI_Scatter(src.data, (endRow - startRow) * cols * 3, MPI_BYTE, localSrc.data,
        (endRow - startRow) * cols * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Start measuring time
    double start = 0.0, end = 0.0;
    if (rank == 0)
    {
        start = MPI_Wtime();
    }

    // Apply adaptive mean filter to local image subset
    adaptiveMeanFilter(localSrc, localDst);

    // Gather filtered image data to root process
    MPI_Gather(localDst.data, (endRow - startRow) * cols * 3, MPI_BYTE, src.data,
        (endRow - startRow) * cols * 3, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Stop measuring time
    if (rank == 0)
    {
        end = MPI_Wtime();

        // Calculate the elapsed time
        double elapsedSeconds = end - start;
        std::cout << "Elapsed time (parallel part only): " << elapsedSeconds << " seconds" << std::endl;

        cv::imshow("src", src); // Show original image
        cv::waitKey(0);
    }

    MPI_Finalize();

    return 0;
}
