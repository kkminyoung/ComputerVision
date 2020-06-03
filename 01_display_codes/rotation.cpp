#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated1, rotated2;

	// Read each image
	input = imread("lena.jpg");

	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// original image
	namedWindow("image");
	imshow("image", input);

	rotated1 = myrotate<Vec3b>(input, 45, "nearest");
	rotated2 = myrotate<Vec3b>(input, 45, "bilinear");

	// nearest interpolation ����� rotated image
	namedWindow("rotated1");
	imshow("rotated1", rotated1);

	// bilinear interpolation ����� rotated image
	namedWindow("rotated2");
	imshow("rotated2", rotated2);

	waitKey(0);

	return 0;
}


template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180;

	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	Mat output = Mat::zeros(sq_row, sq_col, input.type());

	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) {
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
				float x1 = floor(x);
				float x2 = ceil(x);
				float y1 = floor(y);
				float y2 = ceil(y);
				if (!strcmp(opt, "nearest")) {
					float lammda1 = x - x1;
					float lammda2 = y - y1;

					// x�� x1�� �� ������
					if (lammda1 < 0.5) {
						//y�� y1�� �� ������
						if (lammda2 < 0.5) {
							output.at<Vec3b>(i, j) = input.at<Vec3b>(y1, x1);
						}
						// y�� y2�� �� ������
						else {
							output.at<Vec3b>(i, j) = input.at<Vec3b>(y2, x1);
						}
					}
					// x�� x2�� �� ������
					else {
						//y�� y1�� �� ������
						if (lammda2 < 0.5) {
							output.at<Vec3b>(i, j) = input.at<Vec3b>(y1, x2);
						}
						// y�� y2�� �� ������
						else {
							output.at<Vec3b>(i, j) = input.at<Vec3b>(y2, x2);
						}
					}
				}
				else if (!strcmp(opt, "bilinear")) {
					// �� ����(point)
					Vec3b f1 = input.at<Vec3b>(y1, x1);
					Vec3b f2 = input.at<Vec3b>(y1, x2);
					Vec3b f3 = input.at<Vec3b>(y2, x1);
					Vec3b f4 = input.at<Vec3b>(y2, x2);

					// mu, lambda
					float mu = y - y1;
					float lambda = x - x1;

					// 1. 2. 3. ������ ���� ���ϴ� ����
					Vec3b ff1 = mu*f3 + (1-mu)*f1;
					Vec3b ff2 = mu*f4 + (1 - mu)*f2;
					Vec3b ff3 = lambda*ff2 + (1-lambda)*ff1;

					// ������ ff3 ���� output Mat�� �ִ´�
					output.at<Vec3b>(i, j) = ff3;
				}
			}
		}
	}
	return output;
}