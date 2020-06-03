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

Mat Laplacian_filter(const Mat input);

int main() {

	Mat input = imread("C:/Temp/lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);
	output = Laplacian_filter(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", output);

	waitKey(0);

	return 0;
}


Mat Laplacian_filter(const Mat input) {

	Mat kernel;
	float kernelvalue;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Laplacian Filter Kernel N

	// Initialiazing 1 Kernel Matrix with 3x3 size for kernel
	// Fill code to initialize Laplacian filter kernel matrix (Given in the lecture notes)
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
			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;
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

					sum1_r += kernelvalue * ((float)(input.at<C>(tempa, tempb)[0]));
					sum1_g += kernelvalue * ((float)(input.at<C>(tempa, tempb)[1]));
					sum1_b += kernelvalue * ((float)(input.at<C>(tempa, tempb)[2]));

				}

			}
			sum1_r = abs(sum1_r); sum1_g = abs(sum1_g); sum1_b = abs(sum1_b);
			if (sum1_r < 0) sum1_r = 0; if (sum1_r > 255) sum1_r = 255;
			if (sum1_g < 0) sum1_g = 0; if (sum1_g > 255) sum1_g = 255;
			if (sum1_b < 0) sum1_b = 0; if (sum1_b > 255) sum1_b = 255;
			output.at<C>(i, j)[0] = (sum1_r + sum1_g + sum1_b) / 3;
			output.at<C>(i, j)[1] = (sum1_r + sum1_g + sum1_b) / 3;
			output.at<C>(i, j)[2] = (sum1_r + sum1_g + sum1_b) / 3;
		}
	}
	normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1);
	return output;
}