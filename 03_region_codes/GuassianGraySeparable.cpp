#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3
#include <stdio.h>
#include <time.h>

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
Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {

	Mat input = imread("C:/Temp/lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output1;
	Mat output2;

	cvtColor(input, input_gray, CV_RGB2GRAY);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	clock_t start1, end1;
	start1 = clock();
	output1 = gaussianfilter(input_gray, 3, 1, 1, "zero-paddle"); //Boundary process: zero-paddle, mirroring, adjustkernel
	end1 = clock();
	double time1 = (double)(end1 - start1);
	printf("Gaussian 걸린 시간 : %fms\n", time1);

	clock_t start2, end2;
	start2 = clock();
	output2 = fast_gaussianfilter(input_gray, 3, 1, 1, "adjustkernel"); //Boundary process: zero-paddle, mirroring, adjustkernel
	end2 = clock();
	double time2 = (double)(end2 - start2);
	printf("Seperable Gaussian 걸린 시간 : %fms\n", time2);
	

	namedWindow("Gaussian Filter", WINDOW_AUTOSIZE);
	imshow("Gaussian Filter", output1);

	namedWindow("fast Gaussian Filter", WINDOW_AUTOSIZE);
	imshow("fast Gaussian Filter", output2);


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
	kernel_s = Mat::zeros(kernel_size, 1, CV_32F);
	kernel_t = Mat::zeros(1, kernel_size, CV_32F);
	float kernelvalue_s;
	float kernelvalue_t;

	Mat output = Mat::zeros(row, col, input.type());
	Mat temp = Mat::zeros(row, col, input.type());

	denom_s = 0.0;
	denom_t = 0.0;
	for (int a = -n; a <= n; a++) {  
		float value_s = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
		kernel_s.at<float>(a + n,0) = value_s;
		denom_s += value_s;
	}
	for (int b = -n; b <= n; b++) { 
		float value2 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
		kernel_t.at<float>(0,b+n) = value2;
		denom_t += value2;
	}

	for (int a = -n; a <= n; a++) {
		kernel_s.at<float>(a + n, 0) /= denom_s;
	}
	for (int b = -n; b <= n; b++) {
		kernel_t.at<float>(0, b + n) /= denom_t;
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// border 1. zero-padding
			if (!strcmp(opt, "zero-paddle")) {
				float sum1 = 0.0;
				for (int b = -n; b <= n; b++) {
					kernelvalue_t = kernel_t.at<float>(0, b + n);
					if ((j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						sum1 += kernelvalue_t * (float)(input.at<G>(i, j + b));
					}
					temp.at<G>(i, j) = (G)sum1;
				}
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					kernelvalue_s = kernel_s.at<float>(a + n, 0);
					if ((i + a <= row - 1) && (i + a >= 0)) {
						sum2 += kernelvalue_s * (float)(temp.at<G>(i+a, j));
					}
				}
				output.at<G>(i, j) = (G)sum2;
			}

			// border 2. mirroring
			else if (!strcmp(opt, "mirroring")) {
				float sum_s = 0.0;
				float sum_t = 0.0;
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
					sum_t += kernelvalue_t * (float)(input.at<G>(i, tempb));
				}
				temp.at<G>(i, j) = (G)sum_t;

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
					sum_s += kernelvalue_s * temp.at<G>(tempa,j);
				}
				output.at<G>(i, j) = (G)sum_s;
			}

			// border 3. adjust kernel
			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_s = 0.0;
				float sum2_s = 0.0;
				for (int a = -n; a <= n; a++) {
					kernelvalue_s = kernel_s.at<float>(a + n, 0);
					for (int b = -n; b <= n; b++) {
						float sum1_t = 0.0;
						float sum2_t = 0.0;
						kernelvalue_t = kernel_t.at<float>(0,b+n);
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1_t += kernelvalue_t * (float)input.at<G>(i + a, j + b);
							sum2_t += kernelvalue_t;
						}
						sum1_s += kernelvalue_s * sum1_t;
						sum2_s += kernelvalue_s * sum2_t;
					}
				}
				output.at<G>(i, j) = (G)(sum1_s / sum2_s);
			}
		}
	}


	return output;
}

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	float kernelvalue;

	Mat output = Mat::zeros(row, col, input.type());

	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))) - (pow(b, 2) / (2 * pow(sigmaT, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
			printf("%f ", kernel.at<float>(a + n,b+n));
		}
	}


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (!strcmp(opt, "zero-paddle")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
						kernelvalue = kernel.at<float>(a + n, b + n);
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1 += kernelvalue * (float)(input.at<G>(i + a, j + b));
						}
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
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
						sum1 += kernelvalue * (float)(input.at<G>(tempa, tempb));


					}
				}
				output.at<G>(i, j) = (G)sum1;
			}


			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {
						kernelvalue = kernel.at<float>(a + n, b + n);
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += kernelvalue * (float)input.at<G>(i + a, j + b);
							sum2 += kernelvalue;

						}
					}
				}
				output.at<G>(i, j) = (G)(sum1 / sum2);

			}


		}
	}
	return output;
}