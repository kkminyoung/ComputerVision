#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

using namespace cv;

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);
Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt);
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

	// Denoise, using gaussian filter
	Mat Denoised_Gray = Gaussianfilter_Gray(noise_Gray, 3, 10, 10, "zero-padding");
	Mat Denoised_RGB = Gaussianfilter_RGB(noise_RGB, 3, 10, 10, "adjustkernel");

	Mat Denoised_billateral_Gray = Bilateralfilter_Gray(noise_Gray, 3, 10, 10,10, "zero-padding");
	Mat Denoised_billateral_RGB = Bilateralfilter_RGB(noise_RGB, 3, 10, 10,10, "adjustkernel");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (Grayscale)", noise_Gray);

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (RGB)", Denoised_RGB);

	namedWindow("Bilateral Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Bilateral Denoised (Grayscale)", Denoised_billateral_Gray);

	namedWindow("Bilateral Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Bilateral Denoised (RGB)", Denoised_billateral_RGB);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempx;
	int tempy;
	float denom;
	float kernelvalue;

	// Initialiazing Gaussian Kernel Matrix
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	Mat output = Mat::zeros(row, col, input.type());

	denom = 0.0;
	for (int x = -n; x <= n; x++) {  // Denominator in m(s,t)
		for (int y = -n; y <= n; y++) {
			float value1 = exp(-(pow(x, 2) / (2 * pow(sigma_s, 2))) - (pow(y, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<float>(x + n, y + n) = value1;
			denom += value1;
		}
	}

	for (int x = -n; x <= n; x++) {  // Denominator in m(s,t)
		for (int y = -n; y <= n; y++) {
			kernel.at<float>(x + n, y + n) /= denom;
		}
	}

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) {
				float sum1 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						kernelvalue = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							sum1 += kernelvalue * (float)input.at<double>(i + x, j + y);
						}
					}
				}
				output.at<double>(i, j) = (double)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						kernelvalue = kernel.at<float>(x + n, y + n);
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempx = i - x;
						}
						else if (i + x < 0) {
							tempx = -(i + x);
						}
						else {
							tempx = i + x;
						}
						if (j + y > col - 1) {
							tempy = j - y;
						}
						else if (j + y < 0) {
							tempy = -(j + y);
						}
						else {
							tempy = j + y;
						}
						sum1 += kernelvalue * (float)(input.at<double>(tempx, tempy));
					}
				}

			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						kernelvalue = kernel.at<float>(x + n, y + n);
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sum1 += kernelvalue * (float)input.at<double>(i + x, j + y);
							sum2 += kernelvalue;
						}
					}
				}
				output.at<double>(i, j) = (double)sum1 / sum2;
			}

		}
	}

	return output;
}

Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Gaussian Kernel Matrix
	// Fill code to initialize Gaussian filter kernel matrix

	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	Mat output = Mat::zeros(row, col, input.type());

	denom = 0.0;
	for (int a = -n; a <= n; a++) {
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (!strcmp(opt, "zero-padding")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						kernelvalue = kernel.at<float>(a + n, b + n);
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1_r += kernelvalue * (float)(input.at<Vec3d>(i + a, j + b)[0]);
							sum1_g += kernelvalue * (float)(input.at<Vec3d>(i + a, j + b)[1]);
							sum1_b += kernelvalue * (float)(input.at<Vec3d>(i + a, j + b)[2]);
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = sum1_r;
				output.at<Vec3d>(i, j)[1] = sum1_g;
				output.at<Vec3d>(i, j)[2] = sum1_b;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
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
						sum1_r += kernelvalue * (float)(input.at<Vec3d>(tempa, tempb)[0]);
						sum1_g += kernelvalue * (float)(input.at<Vec3d>(tempa, tempb)[1]);
						sum1_b += kernelvalue * (float)(input.at<Vec3d>(tempa, tempb)[2]);
					}
				}
				output.at<Vec3d>(i, j)[0] = sum1_r;
				output.at<Vec3d>(i, j)[1] = sum1_g;
				output.at<Vec3d>(i, j)[2] = sum1_b;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						kernelvalue = kernel.at<float>(a + n, b + n);
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1_r += kernelvalue * (float)(input.at<Vec3d>(i + a, j + b)[0]);
							sum1_g += kernelvalue * (float)(input.at<Vec3d>(i + a, j + b)[1]);
							sum1_b += kernelvalue * (float)(input.at<Vec3d>(i + a, j + b)[2]);
							sum2 += kernelvalue;
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = sum1_r / sum2;
				output.at<Vec3d>(i, j)[1] = sum1_g / sum2;
				output.at<Vec3d>(i, j)[2] = sum1_b / sum2;
			}
		}
	}

	return output;
}

Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float kernelvalue;

	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	float denom;
	// Initialiazing Gaussian Kernel Matrix
	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}

	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) {
				float sum1 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							kernelvalue = kernel.at<float>(x + n, y + n)*exp(-(pow(input.at<double>(i, j) - input.at<double>(i + x, j + y), 2.0)) / 2 * sigma_r);
							sum1 += kernelvalue * (float)(input.at<double>(i + x, j + y));
						}
					}
				}
				output.at<double>(i, j) = (double)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
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
						sum1 += kernelvalue * exp(-(pow(input.at<double>(i, j) - input.at<double>(tempa, tempb), 2.0)) / 2 * sigma_r) * (float)(input.at<double>(tempa, tempb));
					}
				}
				output.at<double>(i, j) = (double)sum1;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							kernelvalue = kernel.at<float>(a + n, b + n)* kernel.at<float>(a + n, b + n)*exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2.0)) / 2 * sigma_r);
							sum1 += kernelvalue * (float)input.at<double>(i + a, j + b);
							sum2 += kernelvalue;
						}
					}
				}
				output.at<double>(i, j) = (double)sum1 / sum2;
			}
		}
	}
	return output;
}

Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Kernel Matrix 
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}

	Mat output = Mat::zeros(row, col, input.type());

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (!strcmp(opt, "zero-padding")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							kernelvalue = kernel.at<float>(a + n, b + n);
							sum1_r += kernelvalue * exp(-(pow((input.at<Vec3d>(i, j)[0] - input.at<Vec3d>(i + a, j + b)[0]), 2.0)) / 2 * sigma_r) * (float)(input.at<Vec3d>(i + a, j + b)[0]);
							sum1_g += kernelvalue * exp(-(pow((input.at<Vec3d>(i, j)[1] - input.at<Vec3d>(i + a, j + b)[1]), 2.0)) / 2 * sigma_r) * (float)(input.at<Vec3d>(i + a, j + b)[1]);
							sum1_b += kernelvalue * exp(-(pow((input.at<Vec3d>(i, j)[2] - input.at<Vec3d>(i + a, j + b)[2]), 2.0)) / 2 * sigma_r) * (float)(input.at<Vec3d>(i + a, j + b)[2]);
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = sum1_r;
				output.at<Vec3d>(i, j)[1] = sum1_g;
				output.at<Vec3d>(i, j)[2] = sum1_b;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
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
						sum1_r += kernelvalue * exp(-(pow((input.at<Vec3d>(i, j)[0] - input.at<Vec3d>(tempa, tempb)[0]), 2.0)) / 2 * sigma_r)* (float)(input.at<Vec3d>(tempa, tempb)[0]);
						sum1_g += kernelvalue * exp(-(pow((input.at<Vec3d>(i, j)[1] - input.at<Vec3d>(tempa, tempb)[1]), 2.0)) / 2 * sigma_r)* (float)(input.at<Vec3d>(tempa, tempb)[1]);
						sum1_b += kernelvalue * exp(-(pow((input.at<Vec3d>(i, j)[2] - input.at<Vec3d>(tempa, tempb)[2]), 2.0)) / 2 * sigma_r)* (float)(input.at<Vec3d>(tempa, tempb)[2]);
					}
				}
				output.at<Vec3d>(i, j)[0] = sum1_r;
				output.at<Vec3d>(i, j)[1] = sum1_g;
				output.at<Vec3d>(i, j)[2] = sum1_b;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							kernelvalue = kernel.at<float>(a + n, b + n);
							sum1_r += kernelvalue * exp(-(pow((input.at<Vec3d>(i, j)[0] - input.at<Vec3d>(i + a, j + b)[0]), 2.0)) / 2 * sigma_r) * (float)(input.at<Vec3d>(i + a, j + b)[0]);
							sum1_g += kernelvalue * exp(-(pow((input.at<Vec3d>(i, j)[1] - input.at<Vec3d>(i + a, j + b)[1]), 2.0)) / 2 * sigma_r) * (float)(input.at<Vec3d>(i + a, j + b)[1]);
							sum1_b += kernelvalue * exp(-(pow((input.at<Vec3d>(i, j)[2] - input.at<Vec3d>(i + a, j + b)[2]), 2.0)) / 2 * sigma_r) * (float)(input.at<Vec3d>(i + a, j + b)[2]);
							sum2 += kernelvalue;
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = sum1_r / sum2;
				output.at<Vec3d>(i, j)[1] = sum1_g / sum2;
				output.at<Vec3d>(i, j)[2] = sum1_b / sum2;
			}
		}
	}
	return output;
}