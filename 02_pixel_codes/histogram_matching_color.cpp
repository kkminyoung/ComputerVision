// Homework - Histogram Matching Color image
// input image를 reference image로

#include "hist_func.h"

void hist_matching(Mat &input, Mat &matched, G *trans_func, float *CDF, float *CDF_ref);

int main() {
	// input, reference
	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat ref = imread("reference.jpg", CV_LOAD_IMAGE_COLOR);
	Mat matched_YUV;


	// RGB -> YUV
	cvtColor(input, matched_YUV, CV_RGB2YUV);
	cvtColor(ref, ref, CV_RGB2YUV);

	// split each channel(Y, U, V)
	Mat channels[3];
	split(matched_YUV, channels);
	Mat Y = channels[0];			// U = channels[1], V = channels[2]

	Mat ref_channels[3];
	split(ref, ref_channels);
	Mat ref_Y = ref_channels[0];

	// PDF or transfer function txt files
	FILE *f_matched_PDF_YUV, *f_PDF_RGB;
	FILE *f_trans_func_ma_YUV;

	float **PDF_RGB = cal_PDF_RGB(input);		// PDF of Input image(RGB) : [L][3]
	float *CDF_YUV = cal_CDF(Y);				// CDF of Y channel image
	float *ref_CDF_YUV = cal_CDF(ref_Y);

	fopen_s(&f_PDF_RGB, "PDF_RGB.txt", "w+");
	fopen_s(&f_matched_PDF_YUV, "matched_PDF_YUV.txt", "w+");
	fopen_s(&f_trans_func_ma_YUV, "trans_func_ma_YUV.txt", "w+");

	G trans_func_ma_YUV[L] = { 0 };			// transfer function

	// histogram matching on Y channel
	hist_matching(Y, Y, trans_func_ma_YUV, CDF_YUV, ref_CDF_YUV);

	// merge Y, U, V channels
	merge(channels, 3, matched_YUV);
	merge(ref_channels, 3, ref);

	// YUV -> RGB (use "CV_YUV2RGB" flag)
	cvtColor(matched_YUV, matched_YUV, CV_YUV2RGB);
	cvtColor(ref, ref, CV_YUV2RGB);

	// matched PDF (YUV)
	float **matched_PDF_YUV = cal_PDF_RGB(matched_YUV);

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF_RGB, "%d\t%f\t%f\t%f\n", i, PDF_RGB[i][0], PDF_RGB[i][1], PDF_RGB[i][2]);
		fprintf(f_matched_PDF_YUV, "%d\t%f\t%f\t%f\n", i, matched_PDF_YUV[i][0], matched_PDF_YUV[i][1], matched_PDF_YUV[i][2]);

		// write transfer functions
		fprintf(f_trans_func_ma_YUV, "%d\t%d\n", i, trans_func_ma_YUV[i]);
	}

	// memory release
	free(PDF_RGB);
	free(CDF_YUV);
	fclose(f_PDF_RGB);
	fclose(f_matched_PDF_YUV);
	fclose(f_trans_func_ma_YUV);

	////////////////////// Show each image ///////////////////////

	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", input);

	namedWindow("reference", WINDOW_AUTOSIZE);
	imshow("reference", ref);

	namedWindow("Matched_YUV", WINDOW_AUTOSIZE);
	imshow("Matched_YUV", matched_YUV);

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