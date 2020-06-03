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

Mat sobelfilter(const Mat input);

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
	output = sobelfilter(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);

	waitKey(0);

	return 0;
}


Mat sobelfilter(const Mat input) {

	Mat Sx;
	Mat Sy;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)
	Sx = Mat::zeros(3, 3, CV_32F);
	Sy = Mat::zeros(3, 3, CV_32F);

	Sx.at<float>(0, 0) = -1.0;
	Sx.at<float>(0, 2) = 1;
	Sx.at<float>(1, 0) = -2;
	Sx.at<float>(1, 2) = 2;
	Sx.at<float>(2, 0) = -1;
	Sx.at<float>(2, 2) = 1;

	Sy.at<float>(0, 0) = -1;
	Sy.at<float>(0, 1) = -2;
	Sy.at<float>(0, 2) = -1;
	Sy.at<float>(2, 0) = 1;
	Sy.at<float>(2, 1) = 2;
	Sy.at<float>(2, 2) = 1;

	float sx;
	float sy;

	int tempa;
	int tempb;

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;
			float sum2_r = 0.0;
			float sum2_g = 0.0;
			float sum2_b = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
					sx = Sx.at<float>(a + n, b + n);
					sy = Sy.at<float>(a + n, b + n);

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

					sum1_r += sx * ((float)(input.at<C>(tempa, tempb)[0]));
					sum1_g += sx * ((float)(input.at<C>(tempa, tempb)[1]));
					sum1_b += sx * ((float)(input.at<C>(tempa, tempb)[2]));


					sum2_r += sy * ((float)(input.at<C>(tempa, tempb)[0]));
					sum2_g += sy * ((float)(input.at<C>(tempa, tempb)[1]));
					sum2_b += sy * ((float)(input.at<C>(tempa, tempb)[2]));

				}

			}

			sum1_r = abs(sum1_r); sum1_g = abs(sum1_g); sum1_b = abs(sum1_b);
			if (sum1_r < 0) sum1_r = 0; if (sum1_r > 255) sum1_r = 255;
			if (sum1_g < 0) sum1_g = 0; if (sum1_g > 255) sum1_g = 255;
			if (sum1_b < 0) sum1_b = 0; if (sum1_b > 255) sum1_b = 255;

			sum2_r = abs(sum2_r); sum2_g = abs(sum2_g); sum2_b = abs(sum2_b);
			if (sum2_r < 0) sum2_r = 0; if (sum2_r > 255) sum2_r = 255;
			if (sum2_g < 0) sum2_g = 0; if (sum2_g > 255) sum2_g = 255;
			if (sum2_b < 0) sum2_b = 0; if (sum2_b > 255) sum2_b = 255;


			float M_r = sqrt(sum1_r*sum1_r + sum2_r * sum2_r);
			float M_g = sqrt(sum1_g*sum1_g + sum2_g * sum2_g);
			float M_b = sqrt(sum1_b*sum1_b + sum2_b * sum2_b);

			float M = (M_r + M_g + M_b) / 3;

			output.at<C>(i, j)[0] = M;
			output.at<C>(i, j)[1] = M;
			output.at<C>(i, j)[2] = M;
		}
	}

	// Visualize the output by multiplying an appropriate constant or mapping the output values into [0,255]
	normalize(output, output, 0, 255, NORM_MINMAX, CV_8UC1);

	return output;
}