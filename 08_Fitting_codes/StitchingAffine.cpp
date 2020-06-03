#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor1(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
int nearestNeighbor2(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);


template <typename T>
Mat cal_affine(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points);

void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha);

int main() {


	Mat input1 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input2 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input1_gray, input2_gray;
	
	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	//resize(input1, input1, Size(input1.cols / 3, input1.rows / 3));
	//resize(input2, input2, Size(input2.cols / 3, input2.rows / 3));

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

	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height))); 
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


	// height(row), width(col) of each image
	const float input1_row = input1.rows;
	const float input1_col = input1.cols;
	const float input2_row = input2.rows;
	const float input2_col = input2.cols;

	// calculate affine Matrix A12, A21
	Mat A21 = cal_affine<float>(srcPoints, dstPoints, (int)srcPoints.size());
	Mat A12 = cal_affine<float>(dstPoints, srcPoints, (int)srcPoints.size());

	printf("%d\n", (int)srcPoints.size());
	std::cout << A21 << std::endl;

	// compute corners (p1, p2, p3, p4)
	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2),
		A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));

	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * input2_col + A21.at<float>(2),
		A21.at<float>(3) * 0 + A21.at<float>(4) * input2_col + A21.at<float>(5));

	Point2f p3(A21.at<float>(0) * input2_row + A21.at<float>(1) * 0 + A21.at<float>(2),
		A21.at<float>(3) * input2_row + A21.at<float>(4) * 0 + A21.at<float>(5));

	Point2f p4(A21.at<float>(0) * input2_row + A21.at<float>(1) * input2_col + A21.at<float>(2),
		A21.at<float>(3) * input2_row + A21.at<float>(4) * input2_col + A21.at<float>(5));

	// for inverse warping
	Point2f p1_(A12.at<float>(0) * 0 + A12.at<float>(1) * 0 + A12.at<float>(2),
		A12.at<float>(3) * 0 + A12.at<float>(4) * 0 + A12.at<float>(5));

	Point2f p2_(A12.at<float>(0) * 0 + A12.at<float>(1) * input1_col + A12.at<float>(2),
		A12.at<float>(3) * 0 + A12.at<float>(4) * input1_col + A12.at<float>(5));

	Point2f p3_(A12.at<float>(0) * input1_row + A12.at<float>(1) * 0 + A12.at<float>(2),
		A12.at<float>(3) * input1_row + A12.at<float>(4) * 0 + A12.at<float>(5));

	Point2f p4_(A12.at<float>(0) * input1_row + A12.at<float>(1) * input1_col + A12.at<float>(2),
		A12.at<float>(3) * input1_row + A12.at<float>(4) * input1_col + A12.at<float>(5));

	// compute boundary for merged image(I_f) 
	int bound_u = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_b = (int)round(std::max(input1_row, std::max(p3.x, p4.x)));
	int bound_l = (int)round(min(0.0f, min(p1.y, p3.y)));
	int bound_r = (int)round(std::max(input1_col, std::max(p2.y, p4.y)));

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
	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));


	// inverse warping with bilinear interplolation
	for (int i = -diff_x_; i < I_f.rows - diff_x_; i++) {
		for (int j = -diff_y_; j < I_f.cols - diff_y_; j++) {
			float x = A12.at<float>(0) * i + A12.at<float>(1) * j + A12.at<float>(2) + diff_x_;
			float y = A12.at<float>(3) * i + A12.at<float>(4) * j + A12.at<float>(5) + diff_y_;

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
	blend_stitching(input1, input2, I_f, bound_l, bound_u, 0.5);



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

	// stitching
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
		Mat v = descriptors.row(i);		// each row of descriptor
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
		Mat v = descriptors.row(i);		// each row of descriptor
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

		// Refine matching points using ratio_based thresholding
		if (ratio_threshold) {
			Mat desc2 = descriptors2.row(nn);
			int second = nearestNeighbor2(desc1, keypoints2, descriptors2);
			Mat desc2_sec = descriptors2.row(second);

			float dist1 = euclidDistance(desc1, desc2);
			float dist2 = euclidDistance(desc1, desc2_sec);

			float r = (float)dist1 / (float)dist2;

			if (r > RATIO_THR) continue;
		}

		// Refine matching points using cross-checking
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
Mat cal_affine(vector<Point2f> srcPoints, vector<Point2f> dstPoints, int number_of_points) {

	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
	Mat b(2 * number_of_points, 1, CV_32F);
	Mat M_trans, temp, affineM;

	// initialize matrix
	for (int i = 0; i < number_of_points; i++) {
		Point2f pt1 = srcPoints[i];
		Point2f pt2 = dstPoints[i];
		M.at<T>(2 * i, 0) = pt1.y;		M.at<T>(2 * i, 1) = pt1.x;		M.at<T>(2 * i, 2) = 1;
		M.at<T>(2 * i + 1, 3) = pt1.y;		M.at<T>(2 * i + 1, 4) = pt1.x;		M.at<T>(2 * i + 1, 5) = 1;
		b.at<T>(2 * i) = pt2.y;		b.at<T>(2 * i + 1) = pt2.x;
	}

	// (M^T * M)^(−1) * M^T * b 
	transpose(M, M_trans);
	invert(M_trans * M, temp);
	affineM = temp * M_trans * b;

	return affineM;
}


void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int bound_l, int bound_u, float alpha) {

	int col = I_f.cols;
	int row = I_f.rows;

	// I2 is already in I_f by inverse warping
	for (int i = 0; i < I1.rows; i++) {
		for (int j = 0; j < I1.cols; j++) {
			bool cond_I2 = I_f.at<Vec3f>(i - bound_u, j - bound_l) != Vec3f(0, 0, 0) ? true : false;

			if (cond_I2)
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = alpha * I1.at<Vec3f>(i, j) + (1 - alpha) * I_f.at<Vec3f>(i - bound_u, j - bound_l);
			else
				I_f.at<Vec3f>(i - bound_u, j - bound_l) = I1.at<Vec3f>(i, j);

		}
	}
}
