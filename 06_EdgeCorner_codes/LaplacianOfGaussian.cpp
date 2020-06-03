// opencv_test.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;


Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
Mat get_Laplacian_Kernel();

Mat Gaussianfilter_gray(const Mat input, int n, double sigma_t, double sigma_s);
Mat Laplacianfilter_gray(const Mat input);

Mat GaussianfilterRGB(const Mat input, int n, double sigma_t, double sigma_s);
Mat LaplacianfilterRGB(const Mat input);
Mat Mirroring(const Mat input, int n);


int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	//Gaussian smoothing parameters
	int window_radius = 2;
	double sigma_t = 2.0;
	double sigma_s = 2.0;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point

	Mat h_f = Gaussianfilter_gray(input_gray, window_radius, sigma_t, sigma_s);	// h(x,y) * f(x,y)
	Mat h_f_rgb = GaussianfilterRGB(input, window_radius, sigma_t, sigma_s);
	Mat Laplacian = Laplacianfilter_gray(h_f);
	Mat Laplacian_rgb = LaplacianfilterRGB(h_f_rgb);

	normalize(Laplacian, Laplacian, 0, 1, CV_MINMAX);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	namedWindow("Gaussian blur", WINDOW_AUTOSIZE);
	imshow("Gaussian blur", h_f);
	namedWindow("Laplacian filter", WINDOW_AUTOSIZE);
	imshow("Laplacian filter", Laplacian);

	namedWindow("input_rgb", WINDOW_AUTOSIZE);
	imshow("input_rgb", input);
	namedWindow("Gaussian blur RGB", WINDOW_AUTOSIZE);
	imshow("Gaussian blur RGB", h_f_rgb);
	namedWindow("Laplacian filter RGB", WINDOW_AUTOSIZE);
	imshow("Laplacian filter RGB", Laplacian_rgb);

	waitKey(0);

	return 0;
}



Mat Gaussianfilter_gray(const Mat input, int n, double sigma_t, double sigma_s) {

	int row = input.rows;
	int col = input.cols;

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);
	Mat output = Mat::zeros(row, col, input.type());

	double kernelvalue;

	//Intermediate data generation for mirroring
	Mat input_mirror = Mirroring(input, n);

	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			//Fill the code
			double sum1 = 0.0;
			for (int a = -n; a <= n; a++) { // for each kernel window
				for (int b = -n; b <= n; b++) {
					kernelvalue = kernel.at<double>(a + n, b + n);
					sum1 += kernelvalue * (double)(input_mirror.at<double>(i + a, j + b));
				}
			}
			output.at<double>(i - n, j - n) = (double)sum1;
		}
	}

	return output;
}

Mat GaussianfilterRGB(const Mat input, int n, double sigma_t, double sigma_s) {

	int row = input.rows;
	int col = input.cols;

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);
	Mat output = Mat::zeros(row, col, input.type());

	double kernelvalue;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						kernelvalue = kernel.at<double>(a + n, b + n);
						sum1_r += kernelvalue * (double)(input.at<Vec3b>(i + a, j + b)[0]);
						sum1_g += kernelvalue * (double)(input.at<Vec3b>(i + a, j + b)[1]);
						sum1_b += kernelvalue * (double)(input.at<Vec3b>(i + a, j + b)[2]);
					}
				}
			}

			output.at<Vec3b>(i, j)[0] = sum1_r;
			output.at<Vec3b>(i, j)[1] = sum1_g;
			output.at<Vec3b>(i, j)[2] = sum1_b;
		}
	}
	return output;
}

Mat Laplacianfilter_gray(const Mat input) {

	int row = input.rows;
	int col = input.cols;

	Mat kernel = get_Laplacian_Kernel();
	Mat output = Mat::zeros(row, col, input.type());

	double kernelvalue;

	int n = 1;
	Mat input_mirror = Mirroring(input, n);

	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			//Fill the code
			double sum1 = 0.0;
			for (int a = -n; a <= n; a++) { // for each kernel window
				for (int b = -n; b <= n; b++) {
					kernelvalue = kernel.at<double>(a + n, b + n);
					sum1 += kernelvalue * (double)(input_mirror.at<double>(i + a, j + b));
				}
			}
			sum1 = abs(sum1);

			output.at<double>(i - n, j - n) = (double)sum1;
		}
	}
	normalize(output, output, 0, 255, NORM_MINMAX);
	return output;
}

Mat LaplacianfilterRGB(const Mat input) {
	int row = input.rows;
	int col = input.cols;
	int tempa, tempb;
	Mat kernel = get_Laplacian_Kernel();
	Mat output = Mat::zeros(row, col, input.type());
	double kernelvalue;
	int n = 1;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
			float sum_r = 0.0;
			float sum_g = 0.0;
			float sum_b = 0.0;

			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					if (i + a > row - 1) {  //mirroring for the border pixels
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else { // in the boundary
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

					kernelvalue = kernel.at<double>(a + n, b + n);
					sum_r += kernelvalue * (double)(input.at<Vec3b>(tempa, tempb)[0]);
					sum_g += kernelvalue * (double)(input.at<Vec3b>(tempa, tempb)[1]);
					sum_b += kernelvalue * (double)(input.at<Vec3b>(tempa, tempb)[2]);
				}
			}
			sum_r = abs(sum_r);
			sum_g = abs(sum_g);
			sum_b = abs(sum_b);

			output.at<Vec3b>(i ,j)[0] = (sum_r + sum_g + sum_b)/3;
			output.at<Vec3b>(i ,j)[1] = (sum_r + sum_g + sum_b)/3;
			output.at<Vec3b>(i ,j)[2] = (sum_r + sum_g + sum_b)/3;

		}
	}

	normalize(output, output, 0, 255, NORM_MINMAX);
	return output;
}



Mat Mirroring(const Mat input, int n)
{
	int row = input.rows;
	int col = input.cols;

	Mat input2 = Mat::zeros(row + 2 * n, col + 2 * n, input.type());
	int row2 = input2.rows;
	int col2 = input2.cols;

	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			input2.at<double>(i, j) = input.at<double>(i - n, j - n);
		}
	}
	for (int i = n; i < row + n; i++) {
		for (int j = 0; j < n; j++) {
			input2.at<double>(i, j) = input2.at<double>(i, 2 * n - j);
		}
		for (int j = col + n; j < col2; j++) {
			input2.at<double>(i, j) = input2.at<double>(i, 2 * col - 2 + 2 * n - j);
		}
	}
	for (int j = 0; j < col2; j++) {
		for (int i = 0; i < n; i++) {
			input2.at<double>(i, j) = input2.at<double>(2 * n - i, j);
		}
		for (int i = row + n; i < row2; i++) {
			input2.at<double>(i, j) = input2.at<double>(2 * row - 2 + 2 * n - i, j);
		}
	}

	return input2;
}


Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize) {

	int kernel_size = (2 * n + 1);
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_64F);
	double kernel_sum = 0.0;

	for (int i = -n; i <= n; i++) {
		for (int j = -n; j <= n; j++) {
			kernel.at<double>(i + n, j + n) = exp(-((i * i) / (2.0*sigma_t * sigma_t) + (j * j) / (2.0*sigma_s * sigma_s)));
			kernel_sum += kernel.at<double>(i + n, j + n);
		}
	}

	if (normalize) {
		for (int i = 0; i < kernel_size; i++)
			for (int j = 0; j < kernel_size; j++)
				kernel.at<double>(i, j) /= kernel_sum;		// normalize
	}

	return kernel;
}

Mat get_Laplacian_Kernel() {

	Mat kernel = Mat::zeros(3, 3, CV_64F);

	kernel.at<double>(0, 1) = 1.0;
	kernel.at<double>(2, 1) = 1.0;
	kernel.at<double>(1, 0) = 1.0;
	kernel.at<double>(1, 2) = 1.0;
	kernel.at<double>(1, 1) = -4.0;

	return kernel;
}