#include <iostream>
#include <opencv2/opencv.hpp>

#define IM_TYPE	CV_8UC3

using namespace cv;

#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#endif

Mat Kmeans_gray1(const Mat input);
Mat Kmeans_gray2(const Mat input, float sigma);
Mat Kmeans_RGB1(const Mat input);
Mat Kmeans_RGB2(const Mat input, float sigma);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	Mat input_gray;
	cvtColor(input, input_gray, CV_RGB2GRAY);

	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", input);

	namedWindow("OriginalGray", WINDOW_AUTOSIZE);
	imshow("OriginalGray", input_gray);

	float sigma = 5;

	Mat output_gray1 = Kmeans_gray1(input_gray); // intensity
	Mat output_gray2 = Kmeans_gray2(input_gray, sigma); // intensity + position

	Mat output_RGB1 = Kmeans_RGB1(input); //color(r, g, b)
	Mat output_RGB2 = Kmeans_RGB2(input, sigma); // color + position(r,g,b,x/sigma,y/sigma)

	namedWindow("Kmeans_gray1", WINDOW_AUTOSIZE);
	imshow("Kmeans_gray1", output_gray1);
	namedWindow("Kmeans_gray2", WINDOW_AUTOSIZE);
	imshow("Kmeans_gray2", output_gray2);

	namedWindow("Kmeans_RGB1", WINDOW_AUTOSIZE);
	imshow("Kmeans_RGB1", output_RGB1);
	namedWindow("Kmeans_RGB2", WINDOW_AUTOSIZE);
	imshow("Kmeans_RGB2", output_RGB2);


	waitKey(0);

	return 0;
}

Mat Kmeans_gray1(const Mat input) {

	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;
	Mat new_image;
	
	Mat samples_gray(input.rows * input.cols, 1, CV_32F);
	for (int x = 0; x < input.rows; x++)
		for (int y = 0; y < input.cols; y++)
			samples_gray.at<float>(x + y * input.rows, 0) = (float)input.at<G>(x, y);

	kmeans(samples_gray, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	new_image = Mat::zeros(input.rows, input.cols, CV_32F);
	for (int x = 0; x < input.rows; x++) {
		for (int y = 0; y < input.cols; y++) {
			int cluster_idx = labels.at<int>(x + y * input.rows, 0);

			//fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
			new_image.at <float>(x, y) = (float)centers.at<float>(cluster_idx, 0) / 255.0;

		}
	}
	return new_image;

}

Mat Kmeans_gray2(const Mat input, float sigma) {

	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;
	Mat new_image;
	
	Mat samples2_gray(input.rows * input.cols, 3, CV_32F);
	for (int x = 0; x < input.rows; x++) {
		for (int y = 0; y < input.cols; y++) {
			// position(i,x/sigma,y/sigma)
			samples2_gray.at<float>(x + y * input.rows, 0) = (float)input.at<G>(x, y) / 255.0; 
			samples2_gray.at<float>(x + y * input.rows, 1) = ((float)x / (float)input.cols) / sigma;
			samples2_gray.at<float>(x + y * input.rows, 2) = ((float)y / (float)input.rows) / sigma;
		}
	}
	kmeans(samples2_gray, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	
	new_image = Mat::zeros(input.rows, input.cols, CV_32F);
	for (int x = 0; x < input.rows; x++)
		for (int y = 0; y < input.cols; y++)
		{
			int cluster_idx = labels.at<int>(x + y * input.rows, 0);
			new_image.at <float>(x, y) = (float)centers.at<float>(cluster_idx, 0);
		}

	return new_image;
}

Mat Kmeans_RGB1(const Mat input) {

	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;

	Mat samples(input.rows * input.cols, 3, CV_32F); 
	for (int x = 0; x < input.rows; x++)
		for (int y = 0; y < input.cols; y++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(x + y * input.rows, z) = input.at<Vec3b>(x, y)[z]; // 

	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	Mat new_image(input.size(), input.type());
	for (int x = 0; x < input.rows; x++)
		for (int y = 0; y < input.cols; y++)
		{
			int cluster_idx = labels.at<int>(x + y * input.rows, 0);

			//Fill code that finds for each pixel of each channel of the output image the intensity of the cluster center.
			new_image.at <Vec3b>(x, y)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at <Vec3b>(x, y)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at <Vec3b>(x, y)[2] = centers.at<float>(cluster_idx, 2);

		}

	return new_image;
}


Mat Kmeans_RGB2(const Mat input, float sigma) {

	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;

	Mat samples2(input.rows * input.cols, 5, CV_32F); 
	for (int x = 0; x < input.rows; x++) {
		for (int y = 0; y < input.cols; y++) {
			for (int z = 0; z < 3; z++) {
				samples2.at<float>(x + y * input.rows, z) = (float)input.at<Vec3b>(x, y)[z] / 255.0; //
			}
			samples2.at<float>(x + y * input.rows, 3) = ((float)x / (float)input.cols) / sigma;
			samples2.at<float>(x + y * input.rows, 4) = ((float)y / (float)input.rows) / sigma;
		}
	}
	kmeans(samples2, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	Mat new_image(input.size(), input.type());
	for (int x = 0; x < input.rows; x++)
		for (int y = 0; y < input.cols; y++)
		{
			int cluster_idx = labels.at<int>(x + y * input.rows, 0);

			new_image.at <Vec3b>(x, y)[0] = centers.at<float>(cluster_idx, 0) * 255;
			new_image.at <Vec3b>(x, y)[1] = centers.at<float>(cluster_idx, 1) * 255;
			new_image.at <Vec3b>(x, y)[2] = centers.at<float>(cluster_idx, 2) * 255;
		}

	return new_image;

}