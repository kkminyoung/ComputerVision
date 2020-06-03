// fast_gaussianfilter : seperable manner

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

Mat fast_gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

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
	output = fast_gaussianfilter(input, 1, 1, 1, "zero-paddle"); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("fast Gaussian Filter", WINDOW_AUTOSIZE);
	imshow("fast Gaussian Filter", output);


	waitKey(0);

	return 0;
}


Mat fast_gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {
	Mat kernel_s;
	Mat kernel_t;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom_s;
	float denom_t;

	// Initialiazing Kernel Matrix 
	kernel_s = Mat::zeros(kernel_size, kernel_size, CV_32F);
	kernel_t = Mat::zeros(kernel_size, kernel_size, CV_32F);
	float kernelvalue_s;
	float kernelvalue_t;

	Mat output = Mat::zeros(row, col, input.type());
	Mat temp = Mat::zeros(row, col, input.type());
	denom_s = 0.0;
	denom_t = 0.0;
	for (int a = -n; a <= n; a++) {  
		float value_s = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
		kernel_s.at<float>(a + n, 0) = value_s;
		denom_s += value_s;
	}
	for (int b = -n; b <= n; b++) { 
		float value2 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
		kernel_t.at<float>(b + n, 0) = value2;
		denom_t += value2;
	}

	for (int a = -n; a <= n; a++) {
		kernel_s.at<float>(a + n, 0) /= denom_s;
	}
	for (int b = -n; b <= n; b++) {
		kernel_t.at<float>(b + n, 0) /= denom_t;
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// border 1. zero-padding
			if (!strcmp(opt, "zero-paddle")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int b = -n; b <= n; b++) {
					kernelvalue_t = kernel_t.at<float>(0, b + n);
					if ((j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						sum1_r += kernelvalue_t * (float)(input.at<C>(i, j + b)[0]);
						sum1_g += kernelvalue_t * (float)(input.at<C>(i, j + b)[1]);
						sum1_b += kernelvalue_t * (float)(input.at<C>(i, j + b)[2]);
					}
				}
				temp.at<C>(i, j)[0] = sum1_r;
				temp.at<C>(i, j)[1] = sum1_g;
				temp.at<C>(i, j)[2] = sum1_b;

				float sum2_r = temp.at<C>(i, j)[0];
				float sum2_g = temp.at<C>(i, j)[1];
				float sum2_b = temp.at<C>(i, j)[2];
				for (int a = -n; a <= n; a++) {
					kernelvalue_s = kernel_s.at<float>(a + n, 0);
					if ((i + a <= row - 1) && (i + a >= 0)) {
						sum2_r += kernelvalue_s * (float)(temp.at<C>(i + a, j)[0]);
						sum2_g += kernelvalue_s * (float)(temp.at<C>(i + a, j)[1]);
						sum2_b += kernelvalue_s * (float)(temp.at<C>(i + a, j)[2]);
					}
				}
				output.at<C>(i, j)[0] = sum2_r;
				output.at<C>(i, j)[1] = sum2_g;
				output.at<C>(i, j)[2] = sum2_b;

			}

			// border 2. mirroring
			else if (!strcmp(opt, "mirroring")) {
				float sum_s_r = 0.0;
				float sum_s_g = 0.0;
				float sum_s_b = 0.0;
				float sum_t_r = 0.0;
				float sum_t_g = 0.0;
				float sum_t_b = 0.0;
				for (int b = -n; b <= n; b++) {
					kernelvalue_t = kernel_t.at<float>(b + n);
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}
					sum_t_r += kernelvalue_t * (float)(input.at<C>(i, tempb)[0]);
					sum_t_g += kernelvalue_t * (float)(input.at<C>(i, tempb)[1]);
					sum_t_b += kernelvalue_t * (float)(input.at<C>(i, tempb)[2]);
				}
				temp.at<C>(i, j)[0] = (G)sum_t_r;
				temp.at<C>(i, j)[1] = (G)sum_t_g;
				temp.at<C>(i, j)[2] = (G)sum_t_b;

				for (int a = -n; a <= n; a++) {
					kernelvalue_s = kernel_s.at<float>(a + n);
					if (i + a > row - 1) {
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					sum_s_r += kernelvalue_s * temp.at<C>(tempa, j)[0];
					sum_s_g += kernelvalue_s * temp.at<C>(tempa, j)[1];
					sum_s_b += kernelvalue_s * temp.at<C>(tempa, j)[2];
				}
				output.at<C>(i, j)[0] = (G)sum_s_r;
				output.at<C>(i, j)[1] = (G)sum_s_g;
				output.at<C>(i, j)[2] = (G)sum_s_b;
			}

			// border 3. adjust kernel
			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_s_r = 0.0;
				float sum1_s_g = 0.0;
				float sum1_s_b = 0.0;

				float sum2_s = 0.0;

				for (int a = -n; a <= n; a++) {
					kernelvalue_s = kernel_s.at<float>(a + n, 0);
					for (int b = -n; b <= n; b++) {
						float sum1_t_r = 0.0;
						float sum1_t_g = 0.0;
						float sum1_t_b = 0.0;
						float sum2_t = 0.0;
						kernelvalue_t = kernel_t.at<float>(b + n, 0);
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1_t_r += kernelvalue_t * (float)input.at<C>(i + a, j + b)[0];
							sum1_t_g += kernelvalue_t * (float)input.at<C>(i + a, j + b)[1];
							sum1_t_b += kernelvalue_t * (float)input.at<C>(i + a, j + b)[2];
							sum2_t += kernelvalue_t;
						}
						sum1_s_r += kernelvalue_s * sum1_t_r;
						sum1_s_g += kernelvalue_s * sum1_t_g;
						sum1_s_b += kernelvalue_s * sum1_t_b;
						sum2_s += kernelvalue_s * sum2_t;
					}
				}
				output.at<C>(i, j)[0] = (sum1_s_r / sum2_s);
				output.at<C>(i, j)[1] = (sum1_s_g / sum2_s);
				output.at<C>(i, j)[2] = (sum1_s_b / sum2_s);
			}
		}
	}


	return output;
}