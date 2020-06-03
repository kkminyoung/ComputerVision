#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

#include <stdio.h>

Mat LaplacianFilter(const Mat input);

int main() {

	Mat input = imread("C:/Temp/lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;

	cvtColor(input, input_gray, CV_RGB2GRAY);



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	output = LaplacianFilter(input_gray); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", output);


	waitKey(0);

	return 0;
}


Mat LaplacianFilter(const Mat input) {

	Mat kernel;
	float kernelvalue;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Laplacian Filter Kernel N

	// Initialiazing 1 Kernel Matrix with 3x3 size for kernel
	// Fill code to initialize Laplacian filter kernel matrix for kernel 
	kernel = Mat::zeros(3, 3, CV_32F);

	kernel.at<float>(0, 1) = 1;
	kernel.at<float>(2, 1) = 1;
	kernel.at<float>(1, 0) = 1;
	kernel.at<float>(1, 2) = 1;
	kernel.at<float>(1, 1) = -4;


	int tempa;
	int tempb;

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					kernelvalue = kernel.at<float>(a + n, b + n);

					if (i + a > row - 1) {  //mirroring for the border pixels
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}
					sum1 += kernelvalue * ((float)(input.at<G>(tempa, tempb)));

				}

			}

			sum1 = abs(sum1);
			if (sum1 < 0) sum1 = 0;
			if (sum1 > 255) sum1 = 255;
			output.at<G>(i, j) = sum1;
		}
	}
	// Visualize the output by multiplying an appropriate constant or mapping the output values into [0,255]
	normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1);
	return output;
}