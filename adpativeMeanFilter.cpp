
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

//����������ڶ�ͼ��ĵ������ؽ����˲������������Ȼ�ȡ��ǰ���ص��ھ����ز�����һ��������Ȼ������������������Ȼ�󣬺�����ȡ��Сֵ�����ֵ����ֵ��
//������������������ֵ�Ƿ�����Сֵ�����ֵ֮�䣬����ǣ���һ����鵱ǰ�����Ƿ�Ҳ�������Χ�ڡ������ǰ�����������Χ�ڣ���ô�ͷ��ص�ǰ����ֵ������ͷ�����ֵ��
//�����ֵ������Сֵ�����ֵ֮�䣬��ô�����������˲����ڲ��ݹ�����������н�һ�����˲���������µ��˲����ڴ�С��������󴰿ڴ�С����ô�����ͻ᷵����ֵ��
//This function is used to filter a single pixel of an image. The function first obtains the neighboring pixels of the current pixel and stores them in a vector, 
//and then sorts this vector. Then, the function obtains the minimum, maximum, and median values. 
//Next, the function will check whether the median is between the minimum and maximum values, and if so, further check whether the current pixel is also within this range. 
//If the current pixel is within this range, then the current pixel value is returned, otherwise the median is returned. 
//If the median is not between the minimum and maximum values, the function will expand the filtering window and recursively call itself for further filtering processing. 
//If the new filtering window size exceeds the maximum window size, the function will return the median.
uchar adaptiveProcess(const Mat& im, int row, int col, int kernelSize, int maxSize)
{
    vector<uchar>pixels;
    for (int a = -kernelSize / 2; a <= kernelSize / 2; a++)
    {
        for (int b = -kernelSize / 2; b <= kernelSize / 2; b++)
        {
            pixels.push_back(im.at<uchar>(row + a, col + b));
        }
    }
    sort(pixels.begin(), pixels.end());
    auto min = pixels[0];
    auto max = pixels[kernelSize * kernelSize - 1];
    auto med = pixels[kernelSize * kernelSize / 2];
    auto zxy = im.at<uchar>(row, col);
    if (med > min && med < max)
    {
        //med����min��С��max����˵����ֵ����һ����������ʱ�������ǰkernel�е�
        //���ĵ�Ҳ�����������ͷ������ĵ��ֵ�������˲���������ĵ����������ŷ�����ֵ�˲���ֵ
        if (zxy > min && zxy < max)
        {
            return zxy;
        }
        else
        {
            return med;
        }
    }
    //�����ֵ��һ�������������󴰿ڣ�Ѱ�Ҳ�����������ֵ�������
    else
    {

        kernelSize += 2;
        if (kernelSize <= maxSize)
        {
            return adaptiveProcess(im, row, col, kernelSize, maxSize);
        }
        else
        {
            return med;
        }
    }
}
/**
 *����Ӧ��ֵ�˲�����ֵ�˲�����Ч�����˲����ڳߴ��Ӱ��ϴ������������ͱ���ͼ���ϸ�ڴ�����ì�ܣ��˲����ڽ�С�����ܺܺõı���ͼ���е�ĳЩϸ�ڣ�
 ���������Ĺ���Ч���Ͳ��Ǻܺã���֮�����ڳߴ�ϴ��нϺõ���������Ч�������ǻ��ͼ�����һ����ģ�������⣬������ֵ�˲��������ֵ�˲����Ĵ��ڳߴ��ǹ̶���С����ģ�
 �Ͳ���ͬʱ���ȥ��ͱ���ͼ���ϸ�ڡ���ʱ��ҪѰ��һ�ָı䣬����Ԥ���趨�õ����������˲��Ĺ����У���̬�ĸı��˲����Ĵ��ڳߴ��С�����������Ӧ��ֵ�˲��� Adaptive Median Filter��
 ���˲��Ĺ����У�����Ӧ��ֵ�˲��������Ԥ���趨�õ��������ı��˲����ڵĳߴ��С��ͬʱ�������һ���������жϵ�ǰ�����ǲ������������������������ֵ�滻����ǰ���أ�
 ���ǣ������ı䡣��ԭ��������˲������ڵ�������ĸ��������������������صĸ���������ֵ�˲��Ͳ��ܺܺõĹ��˵�������
 *@param1:����ͼ��src
 *@param2:���ͼ��dst
 *@param3:minSizeΪ��С��kernelsize
 *@param4: maxSizeΪ����kernlesize
 */
//����������ڶ�����ͼ������˲������������ȶ�����ͼ����б߽����䣬Ȼ�����ͼ���ÿһ�����أ���ÿһ�����ص���adaptiveProcess���������˲�����
//This function is used to filter the entire image. 
//The function first expands the boundary of the input image, then traverses each pixel of the image and calls the adapteProcess function for filtering processing on each pixel
void adpativeMeanFilter(const Mat& src, Mat& dst, int minSize = 3, int maxSize = 7)
{
    //�Ȱ�������kernelsize��makeborder���Է���һ
    //Expand the boundary of the input image, and the new boundary pixel value is a mirror image of the original image boundary pixel value.
    copyMakeBorder(src, dst, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, BORDER_REFLECT);
    int rows = dst.rows;
    int cols = dst.cols;

    for (int j = maxSize / 2; j < rows - maxSize / 2; ++j)
    {
        for (int i = maxSize / 2; i < cols * dst.channels() - maxSize / 2; ++i)
        {
            //����С��kernelsize��ʼ�˲�
            dst.at<uchar>(j, i) = adaptiveProcess(dst, j, i, minSize, maxSize);
        }
    }
}
//�����������һ��ͼ��Ȼ�����adpativeMeanFilter����������ͼ������˲����������ʾԭͼ����˲����ͼ��
//Load an image, and then call the adpativeMeanFilter function to filter the image
int main()
{
    Mat src = imread("hehua.jpg");
    Mat dst;
    adpativeMeanFilter(src, dst);
    imshow("src", src);
    imshow("dst", dst);
    waitKey(0);
    return 0;
}