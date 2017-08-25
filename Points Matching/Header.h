#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"     
#include <fstream>
#include <iostream>
#include "math.h"

using namespace cv;
using namespace std;


vector<int> randvec(int Num)
{
	vector<int> temp;
	int* array = new int[Num];
	for (int i = 0; i < Num; i++)
	{
		array[i] = -1;
	}
	while (temp.size() < 8)
	{
		int n = rand() % (Num + 1);
		if (array[n] == -1)
		{
			temp.push_back(n);
			array[n] = 1;
		}
	}
	return temp;
}

void SelectMatching(vector<DMatch> matchPoints, vector<DMatch> matchPoints2, vector<DMatch>* goodMatchePoints, vector<DMatch>* usingMatchePoints)
{
	//Selecting strong features.
	double minMatch = 1;
	double maxMatch = 0;
	for (int i = 0; i < matchPoints.size(); i++)
	{
		//get the max and min value of all the matches.
		minMatch = minMatch > matchPoints[i].distance ? matchPoints[i].distance : minMatch;
		maxMatch = maxMatch < matchPoints[i].distance ? matchPoints[i].distance : maxMatch;
	}
	cout << "The Best Match is： " << minMatch << endl;
	cout << "The Worst Match is： " << maxMatch << endl;

	for (int i = 0; i < matchPoints.size(); i++)
	{
		if (matchPoints[i].distance < minMatch + (maxMatch - minMatch) / 2.5)
		{
			(*goodMatchePoints).push_back(matchPoints[i]);
		}
	}


	//for (int i = 0; i < matchPoints.size(); i++)
	//{
	//	if (matchPoints[i].distance <= max(2 * minMatch, 0.05))
	//	{
	//		tempMatchePoints.push_back(matchPoints[i]);
	//	}
	//}

	for (int i = 0; i < matchPoints.size(); i++)
	{
		int match = matchPoints[i].trainIdx;
		if (matchPoints2[match].trainIdx == matchPoints[i].queryIdx && matchPoints[i].distance < minMatch + (maxMatch - minMatch) / 3)
		{
			(*usingMatchePoints).push_back(matchPoints[i]);
		}
	}
	std::nth_element((*usingMatchePoints).begin(), (*usingMatchePoints).begin() + 7, (*usingMatchePoints).end());
	(*usingMatchePoints).erase((*usingMatchePoints).begin() + 8, (*usingMatchePoints).end());
}
void FilterMatching(vector<KeyPoint> keyPoint1, vector<KeyPoint> keyPoint2, vector<DMatch>* MatchePoints)
{
	//Get rid of the noise matches.
	double slopesum = 0;
	double lengthsum = 0;
	double avgslope = 0;
	double avglength = 0;
	int num = (*MatchePoints).size();
	int m = 0;
	for (int i = 0; i < num; i++)
	{
		int temp1 = (*MatchePoints)[i].queryIdx;
		int temp2 = (*MatchePoints)[i].trainIdx;
		Point point1 = keyPoint1[temp1].pt;
		Point point2 = keyPoint2[temp2].pt;
		double tempslope = (double)(point2.y - point1.y) / (double)(point2.x + 1920 - point1.x);//1920 here is the width of the image.Change it when applied in Kinect.
		double x = (point2.x + 1920 - point1.x)*(point2.x + 1920 - point1.x);
		double y = (point2.y - point1.y)*(point2.y - point1.y);
		double templength = sqrt(x + y);
		cout << "Match " << i << ", slope= " << tempslope << ", length=" << templength << endl;
		if (tempslope < 0)
		{
			slopesum += tempslope;
			m++;
		}
		lengthsum += templength;
	}
	avgslope = slopesum / m;
	avglength = lengthsum / num;
	cout << "Average: " << "slope= " << avgslope << ", length=" << avglength << endl;
	int nn = 0;
	for (int i = 0; i < num; i++)
	{
		int temp1 = (*MatchePoints)[i - nn].queryIdx;
		int temp2 = (*MatchePoints)[i - nn].trainIdx;
		Point point1 = keyPoint1[temp1].pt;
		Point point2 = keyPoint2[temp2].pt;
		double tempslope = (double)(point2.y - point1.y) / (double)(point2.x + 1920 - point1.x);
		double x = (point2.x + 1920 - point1.x)*(point2.x + 1920 - point1.x);
		double y = (point2.y - point1.y)*(point2.y - point1.y);
		double templength = sqrt(x + y);
		int flag1 = 0;
		int flag2 = 0;
		if (tempslope < avgslope + abs(avgslope) && tempslope > avgslope - abs(avgslope))
			flag1 = 1;
		if (templength<avglength + 0.2*abs(avglength) && templength>avglength - 0.2*abs(avglength))
			flag2 = 1;
		if (flag1 == 0 || flag2 == 0)
		{
			cout << "Delete Num " << (i - nn) << " Values are: length=" << templength << ", slope=" << tempslope << endl;
			(*MatchePoints).erase((*MatchePoints).begin() + (i - nn));
			nn++;
		}
	}
}
void SolveRt(Mat Essential, Mat* Rotation1, Mat* Rotation2, Mat* Transit)
{
	// SVD decompose to get R & T.
	Mat W = Mat::zeros(3, 3, CV_64F);
	Mat WT;
	W.at<double>(0, 1) = -1;
	W.at<double>(1, 0) = 1;
	W.at<double>(2, 2) = 1;
	transpose(W, WT);
	Mat Z = Mat::zeros(3, 3, CV_64F);
	Z.at<double>(0, 1) = 1;
	W.at<double>(1, 0) = -1;
	Mat U, VT, w, E;
	cv::SVD thissvd(Essential);
	U = thissvd.u;
	w = thissvd.w;
	VT = thissvd.vt;
	//cout << D.at<double>(0, 0) << " " << D.at<double>(1, 0) <<" "<< D.at<double>(2, 0) << endl;
	double s = (w.at<double>(0, 0) + w.at<double>(1, 0)) / 2;
	Mat D = Mat::zeros(3, 3, CV_64F);
	D.at<double>(0, 0) = s;
	D.at<double>(1, 1) = s;
	E = U*D*VT;
	cv::SVD fsvd(E);
	U = fsvd.u;
	w = fsvd.w;
	VT = fsvd.vt;
	//cout << w.at<double>(0, 0) << " " << w.at<double>(1, 0) << " " << w.at<double>(2, 0) << endl;
	//Get R
	(*Rotation1) = U*W*VT;
	(*Rotation2) = U*WT*VT;
	if (determinant(*Rotation1) < 0)
		(*Rotation1) = -1 * (*Rotation1);
	if (determinant(*Rotation2) < 0)
		(*Rotation2) = -1 * (*Rotation2);
	//Get T
	Mat UT;
	transpose(U, UT);
	Mat Tx = U*Z*UT;
	(*Transit).at<double>(0, 0) = Tx.at<double>(2, 1);
	(*Transit).at<double>(1, 0) = Tx.at<double>(0, 2);
	(*Transit).at<double>(2, 0) = Tx.at<double>(1, 0);
	//there are four possible solution.
	//(R1,t),(R1,-t),(R2,t),(R2,-t)
}

void function(Mat M, Mat R, Mat T, Mat* temp1, Mat* temp2)
{
	(*temp2) = M*R*T;
	Mat InverM = M.inv();
	(*temp1) = M*R*InverM;

}

void ChooseRT(Mat R1, Mat R2, Mat T, Mat intrinsic, vector<cv::Point3d> Point1, vector<cv::Point3d> Point2)
{
	//Get Coeffcient Matrices.
	Mat H1 = Mat::zeros(3, 3, CV_64F);
	Mat H2 = Mat::zeros(3, 1, CV_64F);
	int N = 0;
	int MatchNum = -1;
	double Loss = 99999;
	int MatchIndex = -1;

	//Validate the results.
	while (N < 4)
	{
		if (N == 0)
			function(intrinsic, R1, T, &H1, &H2);
		if (N == 1)
			function(intrinsic, R1, -T, &H1, &H2);
		if (N == 2)
			function(intrinsic, R2, T, &H1, &H2);
		if (N == 3)
			function(intrinsic, R2, -T, &H1, &H2);


		int tempMatchNum = 0;
		double tempLoss = 0;
		cout << "===================================================" << endl;
		cout << "                  Round " << N << endl;
		cout << "===================================================" << endl;
		for (int i = 0; i < Point1.size(); i++)
		{
			double u = Point1[i].x;
			double v = Point1[i].y;
			double d = Point1[i].z;
			if (d > 500 && d < 5000)
			{
				double z = d / 1000;
				cout << "(u,v,d) information" << endl;
				cout << u << " " << v << " " << d << endl;
				Mat p1 = Mat::zeros(3, 1, CV_64F);
				Mat p2 = Mat::zeros(3, 1, CV_64F);
				p1.at<double>(0, 0) = u;
				p1.at<double>(1, 0) = v;
				p1.at<double>(2, 0) = 1;
				p2 = z*H1*p1 - H2;
				//cout << "This is p2 " << p2.at<double>(0, 0) << " " << p2.at<double>(1, 0) << " " << p2.at<double>(2, 0) << endl;
				double u1 = p2.at<double>(0, 0) / p2.at<double>(2, 0);
				double v1 = p2.at<double>(1, 0) / p2.at<double>(2, 0);

				//cout << "newpoint2 (u1,v1,d1) information" << endl;
				cout << "newpoint2 " << Point2[i].x << " " << Point2[i].y << endl;
				cout << "(u1,v1)  " << u1 << " " << v1 << endl;
				cout << "-----------------------------------" << endl;

				double temp1 = u1 - Point2[i].x;
				double temp2 = v1 - Point2[i].y;
				double weight = 0.3;
				double loss = sqrt(weight*temp1*temp1 + (1 - weight)*temp2*temp2);
				tempLoss += loss;
				if (loss < 100)
					tempMatchNum++;
			}
		}
		cout << "Match Numer is " << tempMatchNum << endl;
		cout << "Average Loss is " << tempLoss / Point1.size() << endl;
		if (tempMatchNum >= MatchNum && tempLoss < Loss)
		{
			Loss = tempLoss;
			MatchNum = tempMatchNum;
			MatchIndex = N;
		}
		N++;
	}
	cout << "Final Decision:" << endl;
	cout << "Case " << MatchIndex << ", Match Number is " << MatchNum << " ,Average Loss is " << Loss / Point1.size() << endl;
	if (MatchIndex == 0)
	{
		cout << "Rotation Matrix" << endl;
		cout << R1.at<double>(0, 0) << " " << R1.at<double>(0, 1) << " " << R1.at<double>(0, 2) << endl;
		cout << R1.at<double>(1, 0) << " " << R1.at<double>(1, 1) << " " << R1.at<double>(1, 2) << endl;
		cout << R1.at<double>(2, 0) << " " << R1.at<double>(2, 1) << " " << R1.at<double>(2, 2) << endl;
		cout << "Translation Vector" << endl;
		cout << T.at<double>(0, 0) << endl;
		cout << T.at<double>(1, 0) << endl;
		cout << T.at<double>(2, 0) << endl;
	}
	if (MatchIndex == 1)
	{
		cout << "Rotation Matrix" << endl;
		cout << R1.at<double>(0, 0) << " " << R1.at<double>(0, 1) << " " << R1.at<double>(0, 2) << endl;
		cout << R1.at<double>(1, 0) << " " << R1.at<double>(1, 1) << " " << R1.at<double>(1, 2) << endl;
		cout << R1.at<double>(2, 0) << " " << R1.at<double>(2, 1) << " " << R1.at<double>(2, 2) << endl;
		cout << "Translation Vector" << endl;
		cout << -T.at<double>(0, 0) << endl;
		cout << -T.at<double>(1, 0) << endl;
		cout << -T.at<double>(2, 0) << endl;
	}
	if (MatchIndex == 2)
	{
		cout << "Rotation Matrix" << endl;
		cout << R2.at<double>(0, 0) << " " << R2.at<double>(0, 1) << " " << R2.at<double>(0, 2) << endl;
		cout << R2.at<double>(1, 0) << " " << R2.at<double>(1, 1) << " " << R2.at<double>(1, 2) << endl;
		cout << R2.at<double>(2, 0) << " " << R2.at<double>(2, 1) << " " << R2.at<double>(2, 2) << endl;
		cout << "Translation Vector" << endl;
		cout << T.at<double>(0, 0) << endl;
		cout << T.at<double>(1, 0) << endl;
		cout << T.at<double>(2, 0) << endl;
	}
	if (MatchIndex == 3)
	{
		cout << "Rotation Matrix" << endl;
		cout << R2.at<double>(0, 0) << " " << R2.at<double>(0, 1) << " " << R2.at<double>(0, 2) << endl;
		cout << R2.at<double>(1, 0) << " " << R2.at<double>(1, 1) << " " << R2.at<double>(1, 2) << endl;
		cout << R2.at<double>(2, 0) << " " << R2.at<double>(2, 1) << " " << R2.at<double>(2, 2) << endl;
		cout << "Translation Vector" << endl;
		cout << -T.at<double>(0, 0) << endl;
		cout << -T.at<double>(1, 0) << endl;
		cout << -T.at<double>(2, 0) << endl;
	}




}
void tofile(Mat intrinsic, vector<cv::Point3d> Point1, vector<cv::Point3d> Point2)
{
	double fx, fy, cx, cy;
	fx = intrinsic.at<double>(0, 0);
	fy = intrinsic.at<double>(1, 1);
	cx = intrinsic.at<double>(0, 2);
	cy = intrinsic.at<double>(1, 2);
	vector<cv::Point3d> temp1(Point1.size()), temp2(Point2.size());
	double maxX = -100;
	double minX = 100;
	double maxY = -100;
	double minY = 100;
	double maxZ = -100;
	double minZ = 100;
	ofstream outmodel;
	outmodel.open("model_1.txt");
	outmodel << Point1.size() << endl;

	for (int i = 0; i < Point1.size(); i++)
	{
		double u = Point1[i].x;
		double v = Point1[i].y;
		double d = Point1[i].z;
		double Z = d / 1000;
		if (Z > maxZ)
			maxZ = Z;
		if (Z < minZ)
			minZ = Z;
		double X = (u - cx)*Z / fx;
		if (X > maxX)
			maxX = X;
		if (X < minX)
			minX = X;
		double Y = (v - cy)*Z / fy;
		if (Y > maxY)
			maxY = Y;
		if (Y < minY)
			minY = Y;
		temp1[i].x = X;
		temp1[i].y = Y;
		temp1[i].z = Z;
	}
	for (int i = 0; i < temp1.size(); i++)
	{
		double NormX = 2 * ((temp1[i].x - minX) / (maxX - minX)) - 1;
		double NormY = 2 * ((temp1[i].y - minY) / (maxY - minY)) - 1;
		double NormZ = 2 * ((temp1[i].z - minZ) / (maxZ - minZ)) - 1;
		outmodel << NormX << " " << NormY << " " << NormZ << endl;
	}
	outmodel.close();

	ofstream outdata;
	outdata.open("data_1.txt");
	outdata << Point2.size() << endl;
	maxX = -100;
	minX = 100;
	maxY = -100;
	minY = 100;
	maxZ = -100;
	minZ = 100;
	for (int i = 0; i < Point2.size(); i++)
	{
		double u = Point2[i].x;
		double v = Point2[i].y;
		double d = Point2[i].z;
		double Z = d / 1000;
		if (Z > maxZ)
			maxZ = Z;
		if (Z < minZ)
			minZ = Z;
		double X = (u - cx)*Z / fx;
		if (X > maxX)
			maxX = X;
		if (X < minX)
			minX = X;
		double Y = (v - cy)*Z / fy;
		if (Y > maxY)
			maxY = Y;
		if (Y < minY)
			minY = Y;
		temp2[i].x = X;
		temp2[i].y = Y;
		temp2[i].z = Z;
	}
	for (int i = 0; i < temp2.size(); i++)
	{
		double NormX = 2 * ((temp2[i].x - minX) / (maxX - minX)) - 1;
		double NormY = 2 * ((temp2[i].y - minY) / (maxY - minY)) - 1;
		double NormZ = 2 * ((temp2[i].z - minZ) / (maxZ - minZ)) - 1;
		outdata << NormX << " " << NormY << " " << NormZ << endl;
	}
	outdata.close();
}

vector<int> RANSC(vector<KeyPoint> keyPoint1, vector<KeyPoint> keyPoint2, vector<DMatch> goodMatchePoints, double treshold)
{
	vector<int> random;
	vector<int> select;
	vector<int> pointIndexes1;
	vector<int> pointIndexes2;
	vector<DMatch> UsingMatches;
	int iternum = 0;
	cv::Mat fundemental;
	vector<cv::Point2f> selPoints1, selPoints2;
	int M = 0;
	while (true)
	{
		pointIndexes1.clear();
		pointIndexes2.clear();
		iternum++;
		int tempM = 0;
		if (goodMatchePoints.size() > 8)
			random = randvec(goodMatchePoints.size());
		else
		{
			for (int i = 0; i < 8; i++)
				random.push_back(i);

		}

		for (int i = 0; i < 8; i++)
		{
			pointIndexes1.push_back(goodMatchePoints[random[i]].queryIdx);
			pointIndexes2.push_back(goodMatchePoints[random[i]].trainIdx);
		}
		KeyPoint::convert(keyPoint1, selPoints1, pointIndexes1);
		KeyPoint::convert(keyPoint2, selPoints2, pointIndexes2);
		// Compute F matrix from 8 matches
		fundemental = cv::findFundamentalMat(
			cv::Mat(selPoints1), // points in first image
			cv::Mat(selPoints2), // points in second image
			CV_FM_8POINT); // 8-point method

		Mat temp1 = Mat::zeros(1, 3, CV_64F);
		Mat temp2 = Mat::zeros(3, 1, CV_64F);
		Mat result;
		//cout << selPoints1[0].x << endl;
		vector<double> results;
		for (int i = 0; i < goodMatchePoints.size(); i++)
		{
			temp1.at<double>(0, 0) = keyPoint1[goodMatchePoints[i].queryIdx].pt.x;
			temp1.at<double>(0, 1) = keyPoint1[goodMatchePoints[i].queryIdx].pt.y;
			temp1.at<double>(0, 2) = 1;
			temp2.at<double>(0, 0) = keyPoint2[goodMatchePoints[i].trainIdx].pt.x;
			temp2.at<double>(1, 0) = keyPoint2[goodMatchePoints[i].trainIdx].pt.y;
			temp2.at<double>(2, 0) = 1;

			result = temp1*fundemental*temp2;
			results.push_back(result.at<double>(0, 0));
			double tempresult = abs(result.at<double>(0, 0));
			if (tempresult < treshold)
				tempM++;
		}
		if (tempM > M)
		{
			M = tempM;
			select = random;
		}
		if (iternum > 100)
		{
			cout << "Maching Numbers: " << M << " over " << goodMatchePoints.size() << endl;
			for (int i = 0; i < 8; i++)
				printf("-- Using Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", random[i], goodMatchePoints[random[i]].queryIdx, goodMatchePoints[random[i]].trainIdx);
			return select;
			break;
		}
	}

}

void Normalize(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_norm, cv::Mat& T)
{
	const int N = points.size();
	if (N == 0)
		return;

	points_norm.resize(N);

	cv::Point2f mean(0, 0);
	for (int i = 0; i < N; ++i)
	{
		mean += points[i];
	}
	mean.x = mean.x / N;
	mean.y = mean.y / N;

	cv::Point2f mean_dev(0, 0);

	for (int i = 0; i < N; ++i)
	{
		points_norm[i] = points[i] - mean;

		mean_dev.x += fabs(points_norm[i].x);
		mean_dev.y += fabs(points_norm[i].y);
	}
	mean_dev.x = mean_dev.x / N;
	mean_dev.y = mean_dev.y / N;

	const float scale_x = 1.0 / mean_dev.x;
	const float scale_y = 1.0 / mean_dev.y;

	for (int i = 0; i < N; i++)
	{
		points_norm[i].x *= scale_x;
		points_norm[i].y *= scale_y;
	}

	T = cv::Mat::eye(3, 3, CV_32F);
	T.at<float>(0, 0) = scale_x;
	T.at<float>(1, 1) = scale_y;
	T.at<float>(0, 2) = -mean.x*scale_x;
	T.at<float>(1, 2) = -mean.y*scale_y;
}
cv::Mat findF(const std::vector<cv::Point2f> pts_prev, const std::vector<cv::Point2f> pts_next)
{
	const int N = pts_prev.size();
	assert(N >= 8);
	std::vector<cv::Point2f> pts_prev_norm;
	std::vector<cv::Point2f> pts_next_norm;
	cv::Mat T1, T2;
	Normalize(pts_prev, pts_prev_norm, T1);
	Normalize(pts_next, pts_next_norm, T2);
	cv::Mat A(N, 9, CV_32F);
	for (int i = 0; i < N; ++i)
	{
		const float u1 = pts_prev_norm[i].x;
		const float v1 = pts_prev_norm[i].y;
		const float u2 = pts_next_norm[i].x;
		const float v2 = pts_next_norm[i].y;
		float* ai = A.ptr<float>(i);

		ai[0] = u2*u1;
		ai[1] = u2*v1;
		ai[2] = u2;
		ai[3] = v2*u1;
		ai[4] = v2*v1;
		ai[5] = v2;
		ai[6] = u1;
		ai[7] = v1;
		ai[8] = 1;
	}
	cv::Mat u, w, vt;

	cv::eigen(A.t()*A, w, vt);
	//cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	cv::Mat Fpre = vt.row(8).reshape(0, 3);

	cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	w.at<float>(2) = 0;

	cv::Mat F_norm = u*cv::Mat::diag(w)*vt;

	cv::Mat F = T2.t()*F_norm*T1;
	float F22 = F.at<float>(2, 2);
	if (fabs(F22) > FLT_EPSILON)
		F /= F22;

	//This part for testing.
	{
		cv::Mat H = T1.t()*F*T2;
		cout << "&^&^&^&^&" << endl;
		cout << H.at<float>(0, 0) << " " << H.at<float>(0, 1) << " " << H.at<float>(0, 2) << endl;
		cout << H.at<float>(1, 0) << " " << H.at<float>(1, 1) << " " << H.at<float>(1, 2) << endl;
		cout << H.at<float>(2, 0) << " " << H.at<float>(2, 1) << " " << H.at<float>(2, 2) << endl;

		Mat result;
		Mat temp1 = Mat::zeros(1, 3, CV_32F);
		Mat temp2 = Mat::zeros(3, 1, CV_32F);
		for (int i = 0; i < N; i++)
		{
			temp1.at<float>(0, 0) = pts_prev_norm[i].x;
			temp1.at<float>(0, 1) = pts_prev_norm[i].y;
			temp1.at<float>(0, 2) = 1;
			temp2.at<float>(0, 0) = pts_next_norm[i].x;
			temp2.at<float>(1, 0) = pts_next_norm[i].y;
			temp2.at<float>(2, 0) = 1;

			result = temp1*F*temp2;
			cout << "This is test in RANSC: " << abs(result.at<float>(0, 0)) << endl;
		}
		
	}

	return F;
}
vector<int> newRANSC(vector<KeyPoint> keyPoint1, vector<KeyPoint> keyPoint2, vector<DMatch> goodMatchePoints, double treshold)
{
	vector<int> random;
	vector<int> select;
	vector<int> pointIndexes1;
	vector<int> pointIndexes2;
	vector<DMatch> UsingMatches;
	int iternum = 0;
	cv::Mat fundemental;
	vector<cv::Point2f> selPoints1, selPoints2;
	int M = 0;
	while (true)
	{
		pointIndexes1.clear();
		pointIndexes2.clear();
		iternum++;
		int tempM = 0;
		if (goodMatchePoints.size() > 8)
			random = randvec(goodMatchePoints.size());
		else
		{
			for (int i = 0; i < 8; i++)
				random.push_back(i);

		}

		for (int i = 0; i < 8; i++)
		{
			pointIndexes1.push_back(goodMatchePoints[random[i]].queryIdx);
			pointIndexes2.push_back(goodMatchePoints[random[i]].trainIdx);
		}
		KeyPoint::convert(keyPoint1, selPoints1, pointIndexes1);
		KeyPoint::convert(keyPoint2, selPoints2, pointIndexes2);
		// Compute F matrix from 8 matches
		fundemental = findF(selPoints1, selPoints2);

		Mat temp1 = Mat::zeros(1, 3, CV_32F);
		Mat temp2 = Mat::zeros(3, 1, CV_32F);
		Mat result;
		//cout << selPoints1[0].x << endl;
		vector<float> results;
		for (int i = 0; i < goodMatchePoints.size(); i++)
		{
			temp1.at<float>(0, 0) = keyPoint1[goodMatchePoints[i].queryIdx].pt.x;
			temp1.at<float>(0, 1) = keyPoint1[goodMatchePoints[i].queryIdx].pt.y;
			temp1.at<float>(0, 2) = 1;
			temp2.at<float>(0, 0) = keyPoint2[goodMatchePoints[i].trainIdx].pt.x;
			temp2.at<float>(1, 0) = keyPoint2[goodMatchePoints[i].trainIdx].pt.y;
			temp2.at<float>(2, 0) = 1;

			result = temp1*fundemental*temp2;
			results.push_back(result.at<float>(0, 0));
			double tempresult = abs(result.at<float>(0, 0));
			if (tempresult < treshold)
				tempM++;
		}
		if (tempM > M)
		{
			M = tempM;
			select = random;
		}
		if (iternum > 100)
		{
			cout << "Maching Numbers: " << M << " over " << goodMatchePoints.size() << endl;
			for (int i = 0; i < 8; i++)
				printf("-- Using Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", random[i], goodMatchePoints[random[i]].queryIdx, goodMatchePoints[random[i]].trainIdx);
			return select;
			break;
		}
	}

}
