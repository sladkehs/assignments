#include "imgproc.h"
#include <iostream>
#include <queue>

namespace IPCVL {
	namespace IMG_PROC {
		void calcHist(cv::InputArray src, double *histogram) {
			cv::Mat inputMat = src.getMat();

			for (int y = 0; y < inputMat.rows; y++) {
				for (int x = 0; x < inputMat.cols; x++) {
					// Todo : histogram을 쌓습니다. 
					/** your code here! **/
					histogram[inputMat.at<uchar>(y, x)]++;
					
					// hint 1 : for loop 를 이용해서 cv::Mat 순회 시 (1채널의 경우) 
					// inputMat.at<uchar>(y, x)와 같이 데이터에 접근할 수 있습니다. 
				}
			}
		}

		void backprojectHistogram(cv::InputArray src_hsv, cv::InputArray face_hsv, cv::OutputArray dst) {
			cv::Mat srcMat = src_hsv.getMat();
			cv::Mat faceMat = face_hsv.getMat();
			dst.create(srcMat.size(), CV_64FC1);
			cv::Mat outputProb = dst.getMat();
			outputProb.setTo(cv::Scalar(0.));

			double model_hist[64][64] = { { 0., } };
			double input_hist[64][64] = { { 0., } };

			// Todo : hs 2차원 히스토그램을 계산하는 함수를 작성합니다. 
			calcHist_hs(srcMat, input_hist);
			calcHist_hs(faceMat, model_hist);

			std::vector<cv::Mat> channels;
			split(srcMat, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			for (int y = 0; y < srcMat.rows; y++) {
				for (int x = 0; x < srcMat.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 
					/** your code here! **/
					int h = UTIL::quantize(mat_h.at<uchar>(y, x));
					int s = UTIL::quantize(mat_s.at<uchar>(y, x));

					outputProb.at<double>(y, x) = UTIL::h_r(model_hist, input_hist, h, s);
					// hint 1 : UTIL::quantize()를 이용해서 srtMat의 값을 양자화합니다. 
					// hint 2 : UTIL::h_r() 함수를 이용해서 outputPorb 값을 계산합니다. 
				}
			}
		}

		void calcHist_hs(cv::InputArray src_hsv, double histogram[][64]) {
			cv::Mat hsv = src_hsv.getMat();
			std::vector<cv::Mat> channels;
			split(hsv, channels);
			cv::Mat mat_h = channels[0];
			cv::Mat mat_s = channels[1];

			// 2차원 히스토그램을 쌓습니다. 
			for (int y = 0; y < hsv.rows; y++) {
				for (int x = 0; x < hsv.cols; x++) {
					// Todo : 양자화된 h,s 값을 얻고 histogram에 값을 더합니다. 

					/** your code here! **/
					int h = UTIL::quantize(mat_h.at<uchar>(y, x));
					int s = UTIL::quantize(mat_s.at<uchar>(y, x));
					histogram[h][s]++;
					// hint 1 : 양자화 시 UTIL::quantize() 함수를 이용해서 mat_h, mat_s의 값을 양자화시킵니다. 
				}
			}

			// 히스토그램을 (hsv.rows * hsv.cols)으로 정규화합니다. 
			for (int j = 0; j < 64; j++) {
				for (int i = 0; i < 64; i++) {
					// Todo : histogram에 있는 값들을 순회하며 (hsv.rows * hsv.cols)으로 정규화합니다. 
					/** your code here! **/
					histogram[j][i] = histogram[j][i] / (hsv.rows * hsv.cols);
				}
			}
		}

		void thresh_binary(cv::InputArray src, cv::OutputArray dst, const int& threshold)
		{
			cv::Mat srcMat = src.getMat();

			dst.create(srcMat.size(), srcMat.type());
			cv::Mat output = dst.getMat();

			for (int y = 0; y < srcMat.rows; y++)
			{
				for (int x = 0; x < srcMat.cols; x++)
				{
					if (srcMat.at<uchar>(y, x) > threshold)
					{
						output.at<uchar>(y, x) = 255;
					}
					else
					{
						output.at<uchar>(y, x) = 0;
					}

				}
			}
		}

		void thresh_otsu(cv::InputArray src, cv::OutputArray dst)
		{
			dst.create(src.size(), CV_8UC1);
			cv::Mat srcMat = src.getMat();
			cv::Mat output = dst.getMat();

			double histogram[256] = { 0, };

			IPCVL::UTIL::calcNormedHist(srcMat, histogram);
			double general_u = IPCVL::UTIL::make_u(srcMat, histogram);

			std::cout << general_u << std::endl;
			double w[256];
			double u0[256];
			double u1[256];
			double v[256];
			int T = 0;
		
			w[0] = histogram[0];
			u0[0] = 0.0;
			v[0] = 0.0;
			double max = 0.;
			for (int t = 1; t < 256; t++)
			{
				w[t] = w[t - 1] + histogram[t];

				if (w[t] == 0.0 || (1 - w[t]) == 0.0)
					continue;

				u0[t] = (w[t - 1] * u0[t - 1] + t * histogram[t]) / w[t];
				u1[t] = (general_u - w[t] * u0[t]) / (1 - w[t]);
				v[t] = w[t] * (1 - w[t])*(u0[t] - u1[t])*(u0[t] - u1[t]);

				std::cout << v[t] << std::endl;
				if (v[t] > max)
				{
					max = v[t];

					T = t;
				}

			}

			for (int y = 0; y < srcMat.rows; y++)
			{
				for (int x = 0; x < srcMat.cols; x++)
				{
					if (srcMat.at<uchar>(y, x) > T)
					{
						output.at<uchar>(y, x) = 255;
					}
					else
					{
						output.at<uchar>(y, x) = 0;
					}

				}
			}
		}

		void flood_fill4(cv::Mat& l, const int& j, const int& i, const int& label)
		{
			if (l.at<int>(j, i) == -1)
			{
				l.at<int>(j, i) = label;
				flood_fill4(l, j, i + 1, label);
				flood_fill4(l, j - 1, i, label);
				flood_fill4(l, j, i - 1, label);
				flood_fill4(l, j + 1, i, label);
			}
		}

		void flood_fill8(cv::Mat& l, const int& j, const int& i, const int& label)
		{
			if (l.at<int>(j, i) == -1)
			{
				l.at<int>(j, i) = label;
				flood_fill4(l, j, i + 1, label);
				flood_fill4(l, j - 1, i + 1, label);
				flood_fill4(l, j - 1, i, label);
				flood_fill4(l, j - 1, i - 1, label);
				flood_fill4(l, j, i - 1, label);
				flood_fill4(l, j + 1, i - 1, label);
				flood_fill4(l, j + 1, i, label);
				flood_fill4(l, j + 1, i + 1, label);
			}
		}

		void efficient_flood_fill4(cv::Mat& l, const int& j, const int& i, const int& label)
		{
			

			std::vector<int> jwapyo;
			std::queue<std::vector<int> > Q;
			
			jwapyo.push_back(j);

			jwapyo.push_back(i);

			Q.push(jwapyo);

			while (!Q.empty())
			{
				std::vector<int> tmp;

				tmp = Q.front();
				Q.pop();

				if (l.at<int>(tmp.front(), tmp.back()) = -1)
				{
					int left, right;
					left = tmp.back();
					right = tmp.back();

					while (l.at<int>(tmp.front(), left - 1) == -1) left--;
					while (l.at<int>(tmp.front(), right + 1) == -1) right++;

					for (int c = left; c <= right; c++)
					{
						l.at<int>(tmp.front(), c) = label;
						if ((l.at<int>(tmp.front() - 1, c) == -1) && (c == left || l.at<int>(tmp.front() - 1, c - 1) != -1))
						{
							std::vector<int> nextjwapyo;
							nextjwapyo.push_back(tmp.front() - 1);
							nextjwapyo.push_back(c);
							Q.push(nextjwapyo);
						}

						if ((l.at<int>(tmp.front() + 1, c) == -1) && (c == left || l.at<int>(tmp.front() + 1, c - 1) != -1))
						{
							std::vector<int> nextjwapyo;
							nextjwapyo.push_back(tmp.front() + 1);
							nextjwapyo.push_back(c);
							Q.push(nextjwapyo);
						}
					}
				}
			}

		}

		void flood_fill(cv::InputArray src, cv::OutputArray dst, const UTIL::CONNECTIVITIES& direction)
		{
			dst.create(src.size(), CV_32SC1);
			cv::Mat srcMat = src.getMat();
			cv::Mat l = dst.getMat();

			IPCVL::UTIL::makeBtoL(srcMat, l);

			/*for (int j = 0; j < l.rows; j++)
			{
				for (int i = 0; i < l.cols; i++)
				{
					std::cout << l.at<int>(j, i);
				}
				std::cout << std::endl;
			}*/

			if (direction == 0)
			{
				int label = 1;
				for (int j = 1; j < l.rows - 1; j++)
				{
					for (int i = 1; i < l.cols - 1; i++)
					{
						if (l.at<int>(j, i) == -1)
						{
							flood_fill4(l, j, i, label);
							label++;
						}
					}
				}
			}
			else if (direction == 1)
			{
				int label = 1;
				for (int j = 1; j < l.rows - 1; j++)
				{
					for (int i = 1; i < l.cols - 1; i++)
					{
						if (l.at<int>(j, i) == -1)
						{
							flood_fill8(l, j, i, label);
							label++;
						}
					}
				}
			}
			else if (direction == 2)
			{
				int label = 1;
				for (int j = 1; j < l.rows - 1; j++)
				{
					for (int i = 1; i < l.cols - 1; i++)
					{
						if (l.at<int>(j, i) == -1)
						{
							efficient_flood_fill4(l, j, i, label);
							label++;
						}
					}
				}
			}

		}

	}  // namespace IMG_PROC
}

