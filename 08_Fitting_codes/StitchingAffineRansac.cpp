#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h> 
#include <stdio.h>
#include <cstdlib>
#include <ctime> 

#define RATIO_THR 0.4
#define S 500
using namespace std;
using namespace cv;

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor1(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
int nearestNeighbor2(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);
template <typename T>
Mat cal_affine(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points, int k);
void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha);
void type2str(int type);

int main() {
	Mat input1 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input1_gray, input2_gray;

	int inlier[500] = { 0, };

	srand((unsigned int)time(NULL));

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	//resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	//resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

	type2str(input1.type());
	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures
		4,		// nOctaveLayers
		0.04,	// contrastThreshold
		10,		// edgeThreshold
		1.6		// sigma
	);

	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	// Create a image for displaying mathing keypoints
	Size size = input2.size();
	Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height))); //input1 이 오른쪽
	input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1;
	Mat descriptors1;

	detector->detect(input1_gray, keypoints1); 
	extractor->compute(input1_gray, keypoints1, descriptors1); 
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	// Detect keypoints
	detector->detect(input2_gray, keypoints2); 
	extractor->compute(input2_gray, keypoints2, descriptors2);

	printf("input2 : %zd keypoints are found.\n", keypoints2.size()); 

	for (int i = 0; i < keypoints1.size(); i++) {
		KeyPoint kp = keypoints1[i];
		kp.pt.x += size.width; 
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	for (int i = 0; i < keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i];
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	bool crossCheck = true;
	bool ratio_threshold = true;
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);

	printf("%zd keypoints are matched.\n", srcPoints.size());

	input1.convertTo(input1, CV_32FC3, 1.0 / 255);
	input2.convertTo(input2, CV_32FC3, 1.0 / 255);


	const float input1_row = input1.rows;
	const float input1_col = input1.cols;
	const float input2_row = input2.rows;
	const float input2_col = input2.cols;


	// calculate affine Matrix A12, A21
	Mat A12[S];

	for (int k = 0; k < S; k++) {
		A12[k] = cal_affine<float>(dstPoints, srcPoints, (int)srcPoints.size(), 4);
	}

	int max = 0;
	int index = 0;

	for (int k = 0; k < S; k++) {
		for (int i = 0; i < (int)dstPoints.size(); i++) {
			Point2f Tp(A12[k].at<float>(0) * dstPoints[i].y + A12[k].at<float>(1) * dstPoints[i].x + A12[k].at<float>(2),
				A12[k].at<float>(3) * dstPoints[i].y + A12[k].at<float>(4) * dstPoints[i].x + A12[k].at<float>(5));

			double dst = (double)sqrt(pow((Tp.x - srcPoints[i].x), 2) + pow((Tp.y - srcPoints[i].y), 2));
			if (dst < 10) {
				printf("%lf\n", dst);
				inlier[k] ++;
			}

		}
		if (inlier[k] > max) {
			max = inlier[k];
			index = k;
		}
	}

	printf("index %d\n", index);


	Mat A12_best = A12[index];

	// for inverse warping
	Point2f p1_(A12_best.at<float>(0) * 0 + A12_best.at<float>(1) * 0 + A12_best.at<float>(2),
		A12_best.at<float>(3) * 0 + A12_best.at<float>(4) * 0 + A12_best.at<float>(5));

	Point2f p2_(A12_best.at<float>(0) * 0 + A12_best.at<float>(1) * input1_col + A12_best.at<float>(2),
		A12_best.at<float>(3) * 0 + A12_best.at<float>(4) * input1_col + A12_best.at<float>(5));

	Point2f p3_(A12_best.at<float>(0) * input1_row + A12_best.at<float>(1) * 0 + A12_best.at<float>(2),
		A12_best.at<float>(3) * input1_row + A12_best.at<float>(4) * 0 + A12_best.at<float>(5));

	Point2f p4_(A12_best.at<float>(0) * input1_row + A12_best.at<float>(1) * input1_col + A12_best.at<float>(2),
		A12_best.at<float>(3) * input1_row + A12_best.at<float>(4) * input1_col + A12_best.at<float>(5));




	// compute boundary for merged image(I_f) 
	int bound_u = 0;
	int bound_b = input1_row;
	int bound_l = 0;
	int bound_r = input1_col;

	// compute boundary for inverse warping
	int bound_u_ = (int)round(min(0.0f, min(p1_.x, p2_.x)));
	int bound_b_ = (int)round(std::max(input2_row, std::max(p3_.x, p4_.x)));
	int bound_l_ = (int)round(min(0.0f, min(p1_.y, p3_.y)));
	int bound_r_ = (int)round(std::max(input2_col, std::max(p2_.y, p4_.y)));

	int diff_x = abs(bound_u);
	int diff_y = abs(bound_l);
	int diff_x_ = abs(bound_u_);
	int diff_y_ = abs(bound_l_);

	// initialize merged image
	Mat I_f(bound_b - bound_u + diff_x_, bound_r - bound_l + diff_y_, CV_32FC3, Scalar(0));


	// inverse warping with bilinear interplolation
	for (int i = -diff_x_; i < I_f.rows - diff_x_; i++) {
		for (int j = -diff_y_; j < I_f.cols - diff_y_; j++) {
			float x = A12_best.at<float>(0) * i + A12_best.at<float>(1) * j + A12_best.at<float>(2) + diff_x_;
			float y = A12_best.at<float>(3) * i + A12_best.at<float>(4) * j + A12_best.at<float>(5) + diff_y_;

			float y1 = floor(y);
			float y2 = ceil(y);
			float x1 = floor(x);
			float x2 = ceil(x);

			float mu = y - y1;
			float lambda = x - x1;

			if (x1 >= 0 && x2 < input2_row && y1 >= 0 && y2 < input2_col)
				I_f.at<Vec3f>(i + diff_x_, j + diff_y_) = lambda * mu * input2.at<Vec3f>(x2, y2) + lambda * (1 - mu) * input2.at<Vec3f>(x2, y1) +
				(1 - lambda) * mu * input2.at<Vec3f>(x1, y2) + (1 - lambda) * (1 - mu) * input2.at<Vec3f>(x1, y1);
		}
	}


	// image stitching with blend
	blend_stitching(input1, input2, I_f, diff_x, diff_y, 0.5);

	// Draw line between nearest neighbor pairs
	for (int i = 0; i < (int)srcPoints.size(); ++i) {
		Point2f pt1 = srcPoints[i];
		Point2f pt2 = dstPoints[i];
		Point2f from = pt1;
		Point2f to = Point(size.width + pt2.x, pt2.y);
		line(matchingImage, from, to, Scalar(0, 0, 255));
	}

	// Display mathing image
	namedWindow("Matching");
	imshow("Matching", matchingImage);

	namedWindow("result");
	imshow("result", I_f);



	waitKey(0);

	return 0;
}

/**
* Calculate euclid distance
*/
double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;
	uchar * a = vec1.ptr(0);
	uchar * b = vec2.ptr(0);
	for (int i = 0; i < dim; i++) {
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(sum);
}

/**
* Find the index of nearest neighbor point from keypoints.
*/
int nearestNeighbor1(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;
	double second = -1;
	int distance = 0;
	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		
		distance = euclidDistance(vec, v);
		if (distance < minDist) {
			second = neighbor;
			minDist = distance;
			neighbor = i;
		}
	}
	return neighbor;
}

//second
int nearestNeighbor2(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;
	double second = -1;
	int distance = 0;
	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		
		distance = euclidDistance(vec, v);
		if (distance < minDist) {
			second = neighbor;
			minDist = distance;
			neighbor = i;
		}
	}
	return second;
}

/**
* Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i);

		int nn = nearestNeighbor1(desc1, keypoints2, descriptors2);

		if (ratio_threshold) {
			Mat desc2 = descriptors2.row(nn);
			int second = nearestNeighbor2(desc1, keypoints2, descriptors2);
			Mat desc2_sec = descriptors2.row(second);

			float dist1 = euclidDistance(desc1, desc2);
			float dist2 = euclidDistance(desc1, desc2_sec);

			float r = (float)dist1 / (float)dist2;

			if (r > RATIO_THR) continue;
		}

		if (crossCheck) {
			Mat desc2 = descriptors2.row(nn);
			int l = nearestNeighbor1(desc2, keypoints1, descriptors1);
			if (i != l)
				continue;
		}
		KeyPoint pt2 = keypoints2[nn];
		srcPoints.push_back(pt1.pt);
		dstPoints.push_back(pt2.pt);
	}
}
template <typename T>
Mat cal_affine(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points, int k) {

	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
	Mat b(2 * number_of_points, 1, CV_32F);
	Mat M_trans, temp, affineM;

	// initialize matrix
	int arr[5] = { 0,0,0,0 };
	int num_check[500] = { 0, };

	for (int i = 0; i < k; i++) {
		int num = rand() % number_of_points;
		printf("random number %d\n", num);
		if (num_check[num] == 0) {
			num_check[num] = 1;
			arr[i] = num;
		}
		else i--;
	}

	for (int i = 0; i < k; i++) {
		Point2f pt1 = srcPoints[arr[i]];
		Point2f pt2 = dstPoints[arr[i]];
		M.at<T>(2 * i, 0) = pt1.y;		M.at<T>(2 * i, 1) = pt1.x;		M.at<T>(2 * i, 2) = 1;
		M.at<T>(2 * i + 1, 3) = pt1.y;		M.at<T>(2 * i + 1, 4) = pt1.x;		M.at<T>(2 * i + 1, 5) = 1;
		b.at<T>(2 * i) = pt2.y;		b.at<T>(2 * i + 1) = pt2.x;
	}

	// (M^T * M)^(−1) * M^T * b ( * : Matrix multiplication)
	transpose(M, M_trans);
	invert(M_trans * M, temp);
	affineM = temp * M_trans * b;



	return affineM;
}



void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha) {

	printf("blend\n");
	int bound_x = I1.rows + diff_x;
	int bound_y = I1.cols + diff_y;

	int col = I_f.cols;
	int row = I_f.rows;

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++) {
			// for check validation of I1 & I2
			bool cond1 = (i < bound_x && i > diff_x) && (j < bound_y && j > diff_y) ? true : false;
			bool cond2 = I_f.at<Vec3f>(i, j) != Vec3f(0, 0, 0) ? true : false;

			// I2 is already in I_f by inverse warping
			// So, It is not necessary to check that only I2 is valid
			// if both are valid
			if (cond1 && cond2) {
				I_f.at<Vec3f>(i, j) = alpha * I1.at<Vec3f>(i - diff_x, j - diff_y) + (1 - alpha) * I_f.at<Vec3f>(i, j);

			}

			// only I1 is valid
			else if (cond1) {
				I_f.at<Vec3f>(i + diff_x, j + diff_y) = I1.at<Vec3f>(i, j);
				//I_f.at<Vec3f>(i, j) = 0;
			}
		}
	}
	printf("blend\n");
}
//type2str(input.type());

void type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	printf("Matrix: %s \n", r.c_str());
}

