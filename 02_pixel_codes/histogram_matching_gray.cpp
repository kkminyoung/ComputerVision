// Homework - Histogram Matching gray image

#include "hist_func.h"

void hist_matching(Mat &input, Mat &matched, G *trans_func, float *CDF, float *CDF_ref);

int main() {
	// input, reference
	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat ref = imread("reference.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ref_gray;
	Mat matched_gray;

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
	cvtColor(ref, ref_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
	matched_gray = input_gray.clone();

	// PDF or transfer function txt files
	FILE *f_PDF;
	FILE *f_matched_PDF_gray;
	FILE *f_trans_func_ma_gray;

	fopen_s(&f_PDF, "PDF.txt", "w+");
	fopen_s(&f_matched_PDF_gray, "matched_PDF_gray.txt", "w+");
	fopen_s(&f_trans_func_ma_gray, "trans_func_ma_gray.txt", "w+");

	float *PDF = cal_PDF(input_gray);		// PDF of Input image(RGB) : [L][3]
	float *CDF = cal_CDF(input_gray);				// CDF of Y channel image
	float *ref_CDF = cal_CDF(ref_gray);

	G trans_func_ma_gray[L] = { 0 };			// transfer function

	// histogram matching on Y channel
	hist_matching(input_gray, matched_gray, trans_func_ma_gray, CDF, ref_CDF);

	// matched PDF (YUV)
	float *matched_PDF_gray = cal_PDF(matched_gray);

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF, "%d\t%f\n", i, PDF[i]);
		fprintf(f_matched_PDF_gray, "%d\t%f\n", i, matched_PDF_gray[i]);

		// write transfer functions
		fprintf(f_trans_func_ma_gray, "%d\t%d\n", i, trans_func_ma_gray[i]);
	}

	// memory release
	free(PDF);
	free(CDF);
	fclose(f_PDF);
	fclose(f_matched_PDF_gray);
	fclose(f_trans_func_ma_gray);

	////////////////////// Show each image ///////////////////////

	namedWindow("input_gray", WINDOW_AUTOSIZE);
	imshow("input_gray", input_gray);

	namedWindow("reference_gray", WINDOW_AUTOSIZE);
	imshow("reference_gray", ref_gray);

	namedWindow("Matched_gray", WINDOW_AUTOSIZE);
	imshow("Matched_gray", matched_gray);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram matching
void hist_matching(Mat &input, Mat &matched, G *trans_func, float *CDF, float *ref_CDF) {

	// 1. compute transfer function s=T(r)   * r: input image
	G trans_func_T[L] = { 0 };
	for (int i = 0; i < L; i++)
		trans_func_T[i] = (G)((L - 1) * CDF[i]);

	// 2. compute transfer function s=G(z)   * z: reference image
	G trans_func_G[L] = { 0 };
	for (int i = 0; i < L; i++) {
		trans_func_G[i] = (G)((L - 1)*ref_CDF[i]);
	}


	// 3. Apply the intensity mapping from r to z
	// z = G^(-1)(s)
	G trans_func_G_inv[L] = { 0 };

	int s;
	for (int i = 0; i < L; i++) {
		s = trans_func_G[i];
		if (trans_func_G_inv[s] == NULL) {
			trans_func_G_inv[s] = i;
		}
	}

	s = 0;
	while (s <= L) {
		if (trans_func_G_inv[s] == NULL) {
			if (trans_func_G_inv[s - 1] == 255)
				trans_func_G_inv[s] = 255;
			else
				trans_func_G_inv[s] = trans_func_G_inv[s - 1] + 1;
		}
		s++;
	}

	for (int i = 0; i < L; i++)
		trans_func[i] = trans_func_G_inv[trans_func_T[i]];

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			matched.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}