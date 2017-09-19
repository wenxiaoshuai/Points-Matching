﻿#include "Header.h"

using namespace cv;
using namespace std;

//function for RANSC.
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

//Selecting good matches based on double match and Surf descriptor distance.
void SelectMatching(vector<DMatch> matchPoints, vector<DMatch> matchPoints2, vector<DMatch>* goodMatchePoints, vector<DMatch>* usingMatchePoints)
{
	//Selecting strong features.
	double minMatch = 999999;
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
		int match = matchPoints[i].trainIdx;
		if (matchPoints2[match].trainIdx == matchPoints[i].queryIdx&&matchPoints[i].distance < minMatch + (maxMatch - minMatch) / 2)
		{
			(*goodMatchePoints).push_back(matchPoints[i]);
		}
	}


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
// A filter to remove poor matches.
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
//Decompose R and T from Essential Matrix.
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
	Z.at<double>(1, 0) = -1;

	Mat U, VT, w, E;
	cv::SVD thissvd(Essential);
	U = thissvd.u;
	w = thissvd.w;
	VT = thissvd.vt;

	//cout << w.at<double>(0, 0) << " " << w.at<double>(1, 0) <<" "<< w.at<double>(2, 0) << endl;
	double s = (w.at<double>(0, 0) + w.at<double>(1, 0)) / 2;
	Mat D = Mat::zeros(3, 3, CV_64F);
	D.at<double>(0, 0) = s;
	D.at<double>(1, 1) = s;
	E = U*D*VT;
	//cv::SVD fsvd(E);
	//U = fsvd.u;
	//w = fsvd.w;
	//VT = fsvd.vt;

	cv::SVDecomp(E, w, U, VT, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	cout << w.at<double>(0, 0) << " " << w.at<double>(1, 0) << " " << w.at<double>(2, 0) << endl;
	//Get R
	(*Rotation1) = U*W*VT;
	(*Rotation2) = U*WT*VT;
	//cout << "DET= " << determinant(*Rotation1) << endl;
	if (determinant(*Rotation1) < 0)
		(*Rotation1) = -1 * (*Rotation1);
	if (determinant(*Rotation2) < 0)
		(*Rotation2) = -1 * (*Rotation2);
	//Get T
	Mat UT;
	transpose(U, UT);
	Mat Tx = U*Z*UT;

	(*Transit) = U.col(2);
	cout << "Check Point Here:|T|= " << (*Transit).at<double>(0, 0)*(*Transit).at<double>(0, 0) + (*Transit).at<double>(1, 0)*(*Transit).at<double>(1, 0) + (*Transit).at<double>(2, 0)*(*Transit).at<double>(2, 0) << endl;
	//there are four possible solution.
	//(R1,t),(R1,-t),(R2,t),(R2,-t)
}

void function(Mat M, Mat R, Mat T, Mat* temp1, Mat* temp2)
{
	(*temp2) = M*R*T;
	Mat InverM = M.inv();
	(*temp1) = M*R*InverM;

}

//Chose R&T with 2D points.(technically, the T here is a*t.(t is the real transit vector)).
int ChooseRT1(Mat R1, Mat R2, Mat T, Mat intrinsic, vector<cv::Point3d> Point1, vector<cv::Point3d> Point2)
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
	cout << "Case " << MatchIndex << ", Match Number is " << MatchNum << " over " << Point1.size() << " ,Average Loss is " << Loss / Point1.size() << endl;
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
	return MatchIndex;

}
//Calculate the distance between two vector.
double getDis(Mat a, int n)
{
	double distance = 0;
	for (int i = 0; i < a.rows; i++)
	{
		distance += (a.at<double>(i, n) - a.at<double>(i, n + 1))*(a.at<double>(i, n) - a.at<double>(i, n + 1));
	}
	return distance;
}
//Get the Maximum distance within the vector sets.
double getMaxDis(Mat a, int size)
{
	int j = 0;
	double dis = 0;
	while (j < size - 1)
	{
		for (int i = j; i < size - 1; i++)
		{
			double tempdis = getDis(a, i);
			if (tempdis > dis)
			{
				dis = tempdis;
			}
		}
		j++;
	}
	return dis;
}
//Using 3D points set to calculate the real R and t.
void ChooseRT2(Mat R1, Mat R2, Mat* T, Mat intrinsic, vector<cv::Point3d> Point1, vector<cv::Point3d> Point2)
{
	//Get Coeffcient Matrices.
	Mat R = Mat::zeros(3, 3, CV_64F);
	Mat t = Mat::zeros(3, 1, CV_64F);
	int N = 0;
	int MatchIndex = -1;

	double fx = intrinsic.at<double>(0, 0);
	double fy = intrinsic.at<double>(1, 1);
	double cx = intrinsic.at<double>(0, 2);
	double cy = intrinsic.at<double>(1, 2);
	//Validate the results.
	Mat tempT0 = Mat::zeros(3, Point1.size(), CV_64F);;
	Mat tempT1 = Mat::zeros(3, Point1.size(), CV_64F);;
	for (N = 0; N < 2; N++)
	{
		if (N == 0)
			R = R1;
		if (N == 1)
			R = R2;

		cout << "===================================================" << endl;
		cout << "                  Round " << N << endl;
		cout << "===================================================" << endl;
		for (int i = 0; i < Point1.size(); i++)
		{
			double u1 = Point1[i].x;
			double v1 = Point1[i].y;
			double d1 = Point1[i].z;
			double u2 = Point2[i].x;
			double v2 = Point2[i].y;
			double d2 = Point2[i].z;

			double z1 = d1 / 1000;
			double x1 = z1*(u1 - cx) / fx;
			double y1 = z1*(v1 - cy) / fy;
			double z2 = d2 / 1000;
			double x2 = z2*(u1 - cx) / fx;
			double y2 = z2*(v1 - cy) / fy;

			Mat P1 = Mat::zeros(3, 1, CV_64F);
			Mat P2 = Mat::zeros(3, 1, CV_64F);

			P1.at<double>(0, 0) = x1;
			P1.at<double>(1, 0) = y1;
			P1.at<double>(2, 0) = z1;

			P2.at<double>(0, 0) = x2;
			P2.at<double>(1, 0) = y2;
			P2.at<double>(2, 0) = z2;
			if (N == 0)
				tempT0.col(i) = P2 - R*P1;
			if (N == 1)
				tempT1.col(i) = P2 - R*P1;

		}
	}
	int index = -1;
	double dis0 = getMaxDis(tempT0, tempT0.cols);
	double dis1 = getMaxDis(tempT1, tempT1.cols);
	if (dis0 < dis1)
	{
		cout << "choose R1" << endl;
		cout << "Rotation Matrix" << endl;
		cout << R1.at<double>(0, 0) << " " << R1.at<double>(0, 1) << " " << R1.at<double>(0, 2) << endl;
		cout << R1.at<double>(1, 0) << " " << R1.at<double>(1, 1) << " " << R1.at<double>(1, 2) << endl;
		cout << R1.at<double>(2, 0) << " " << R1.at<double>(2, 1) << " " << R1.at<double>(2, 2) << endl;
		for (int i = 0; i < tempT0.cols; i++)
		{
			t += tempT0.col(i);
		}
		t = t / tempT0.cols;
		(*T) = t;
		cout << "Transit Vector" << endl;
		cout << (*T).at<double>(0, 0) << endl;
		cout << (*T).at<double>(1, 0) << endl;
		cout << (*T).at<double>(2, 0) << endl;
	}
	else
	{
		cout << "Choose R2" << endl;
		cout << "Rotation Matrix" << endl;
		cout << R2.at<double>(0, 0) << " " << R2.at<double>(0, 1) << " " << R2.at<double>(0, 2) << endl;
		cout << R2.at<double>(1, 0) << " " << R2.at<double>(1, 1) << " " << R2.at<double>(1, 2) << endl;
		cout << R2.at<double>(2, 0) << " " << R2.at<double>(2, 1) << " " << R2.at<double>(2, 2) << endl;
		for (int i = 0; i < tempT1.cols; i++)
		{
			t += tempT1.col(i);
			//cout << "TempT1 " << i << "  : " << tempT1.at<double>(0, i) << " " << tempT1.at<double>(1, i) << " " << tempT1.at<double>(2, i) << endl;
		}
		t = t / tempT1.cols;
		(*T) = t;
		cout << "Transit Vector" << endl;
		cout << (*T).at<double>(0, 0) << endl;
		cout << (*T).at<double>(1, 0) << endl;
		cout << (*T).at<double>(2, 0) << endl;
	}

}
//Validating the results from SolvePnp routine.
void ValidatePnp(Mat R, Mat t, Mat intrinsic, vector<cv::Point3d> Point1, vector<cv::Point3d> Point2)
{
	double fx = intrinsic.at<double>(0, 0);
	double fy = intrinsic.at<double>(1, 1);
	double cx = intrinsic.at<double>(0, 2);
	double cy = intrinsic.at<double>(1, 2);
	double error = 0;
	cout << "===================================================" << endl;
	cout << "                  Validate Results " << endl;
	cout << "===================================================" << endl;
	for (int i = 0; i < Point1.size(); i++)
	{
		double u = Point1[i].x;
		double v = Point1[i].y;
		double d = Point1[i].z;
		if (d > 500 && d < 5000)
		{
			double z = d / 1000;
			double x = z*(u - cx) / fx;
			double y = z*(v - cy) / fy;
			Mat P1 = Mat::zeros(3, 1, CV_64F);
			Mat P2 = Mat::zeros(3, 1, CV_64F);
			P1.at<double>(0, 0) = x;
			P1.at<double>(1, 0) = y;
			P1.at<double>(2, 0) = z;
			P2 = R*P1 + t;
			//cout << "This is p2 " << p2.at<double>(0, 0) << " " << p2.at<double>(1, 0) << " " << p2.at<double>(2, 0) << endl;
			double x2 = P2.at<double>(0, 0);
			double y2 = P2.at<double>(1, 0);
			double z2 = P2.at<double>(2, 0);

			cout << "P2 information" << endl;
			cout << "prex= " << x2 << " prey= " << y2 << " prez= " << z2 << endl;
			double tempu = Point2[i].x;
			double tempv = Point2[i].y;
			double tempd = Point2[i].z;
			double tempz = tempd / 1000;
			double tempx = tempz*(tempu - cx) / fx;
			double tempy = tempz*(tempv - cy) / fy;
			cout << "x= " << tempx << " y= " << tempy << " z= " << tempz << endl;
			double sqerror = (x2 - tempx)*(x2 - tempx) + (y2 - tempy)*(y2 - tempy) + (z2 - tempz)*(z2 - tempz);
			error += sqerror;
			cout << "-----------------------------------" << endl;
		}
	}
	cout << "Squared Error= " << error << endl;
}
//Write 3D points sets to file in order to porcessing the GO-ICP Algorithm.
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
//Normalizing the data.
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

	const double scale_x = 1.0 / mean_dev.x;
	const double scale_y = 1.0 / mean_dev.y;

	for (int i = 0; i < N; i++)
	{
		points_norm[i].x *= scale_x;
		points_norm[i].y *= scale_y;
	}

	T = cv::Mat::eye(3, 3, CV_64F);
	T.at<double>(0, 0) = scale_x;
	T.at<double>(1, 1) = scale_y;
	T.at<double>(0, 2) = -mean.x*scale_x;
	T.at<double>(1, 2) = -mean.y*scale_y;
}
//2D based 8 points algorithm.
cv::Mat findF2D(const std::vector<cv::Point2f> pts_prev, const std::vector<cv::Point2f> pts_next)
{
	const int N = pts_prev.size();
	assert(N >= 8);
	std::vector<cv::Point2f> pts_prev_norm;
	std::vector<cv::Point2f> pts_next_norm;
	cv::Mat T1, T2;
	//Normalize(pts_prev, pts_prev_norm, T1);
	//Normalize(pts_next, pts_next_norm, T2);
	cv::Mat A(N, 9, CV_64F);
	for (int i = 0; i < N; ++i)
	{
		/*	const float u1 = pts_prev_norm[i].x;
		const float v1 = pts_prev_norm[i].y;
		const float u2 = pts_next_norm[i].x;
		const float v2 = pts_next_norm[i].y;*/
		const double u1 = pts_prev[i].x;
		const double v1 = pts_prev[i].y;
		const double u2 = pts_next[i].x;
		const double v2 = pts_next[i].y;
		double* ai = A.ptr<double>(i);

		ai[0] = u2*u1;
		ai[1] = u1*v2;
		ai[2] = u1;
		ai[3] = v1*u2;
		ai[4] = v1*v2;
		ai[5] = v1;
		ai[6] = u2;
		ai[7] = v2;
		ai[8] = 1;
	}
	cv::Mat u, w, vt;

	//cv::eigen(A.t()*A, w, vt);
	cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	cv::Mat Fpre = vt.row(8).reshape(0, 3);

	cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	//cout << "W(1) = " << w.at<double>(0) << endl;
	//cout << "W(2) = " << w.at<double>(1) << endl;
	//cout << "W(3) = " << w.at<double>(2) << endl;
	w.at<double>(2) = 0;

	cv::Mat F_norm = u*cv::Mat::diag(w)*vt;

	//This part for testing.
	{
		Mat result;
		Mat temp1 = Mat::zeros(1, 3, CV_64F);
		Mat temp2 = Mat::zeros(3, 1, CV_64F);
		for (int i = 0; i < N; i++)
		{
			temp1.at<double>(0, 0) = pts_prev[i].x;
			temp1.at<double>(0, 1) = pts_prev[i].y;
			temp1.at<double>(0, 2) = 1;
			temp2.at<double>(0, 0) = pts_next[i].x;
			temp2.at<double>(1, 0) = pts_next[i].y;
			temp2.at<double>(2, 0) = 1;

			result = temp1*Fpre*temp2;
			//cout << "This is test in RANSC: " << abs(result.at<double>(0, 0)) << endl;
		}
	}

	return F_norm;
}
//3D based 8 points algorithm.
cv::Mat findF3D(const std::vector<cv::Point3d> pts_prev, const std::vector<cv::Point3d> pts_next, Mat intrinsic)
{
	const int N = pts_prev.size();
	assert(N >= 8);
	std::vector<cv::Point3d> prev(N);
	std::vector<cv::Point3d> next(N);

	//cv::Mat T1, T2;
	//Normalize(pts_prev, pts_prev_norm, T1);
	//Normalize(pts_next, pts_next_norm, T2);
	double fx = intrinsic.at<double>(0, 0);
	double fy = intrinsic.at<double>(1, 1);
	double cx = intrinsic.at<double>(0, 2);
	double cy = intrinsic.at<double>(1, 2);

	for (int i = 0; i < pts_prev.size(); i++)
	{
		const double u1 = pts_prev[i].x;
		const double v1 = pts_prev[i].y;
		const double d1 = pts_prev[i].z;
		const double u2 = pts_next[i].x;
		const double v2 = pts_next[i].y;
		const double d2 = pts_next[i].z;

		double z1 = d1 / 1000;
		double x1 = z1*(u1 - cx) / fx;
		double y1 = z1*(v1 - cy) / fy;
		double z2 = d2 / 1000;
		double x2 = z2*(u2 - cx) / fx;
		double y2 = z2*(v2 - cy) / fy;

		prev[i].x = x1;
		prev[i].y = y1;
		prev[i].z = z1;
		next[i].x = x2;
		next[i].y = y2;
		next[i].z = z2;

	}
	cv::Mat A(N, 9, CV_64F);
	for (int i = 0; i < N; ++i)
	{

		const double x1 = prev[i].x;
		const double y1 = prev[i].y;
		const double z1 = prev[i].z;
		const double x2 = next[i].x;
		const double y2 = next[i].y;
		const double z2 = next[i].z;

		double* ai = A.ptr<double>(i);

		ai[0] = x1*x2;
		ai[1] = y1*x2;
		ai[2] = z1*x2;
		ai[3] = x1*y2;
		ai[4] = y1*y2;
		ai[5] = z1*y2;
		ai[6] = x1*z2;
		ai[7] = y1*z2;
		ai[8] = z1*z2;
	}
	cv::Mat u, w, vt;

	//cv::eigen(A.t()*A, w, vt);
	cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	cv::Mat Fpre = vt.row(8).reshape(0, 3);

	cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	return Fpre;
}
cv::Mat findF3DF(const std::vector<cv::Point3d> pts_prev, const std::vector<cv::Point3d> pts_next, Mat intrinsic)
{
	const int N = pts_prev.size();
	assert(N >= 8);
	std::vector<cv::Point3d> prev(N);
	std::vector<cv::Point3d> next(N);

	//cv::Mat T1, T2;
	//Normalize(pts_prev, pts_prev_norm, T1);
	//Normalize(pts_next, pts_next_norm, T2);
	double fx = intrinsic.at<double>(0, 0);
	double fy = intrinsic.at<double>(1, 1);
	double cx = intrinsic.at<double>(0, 2);
	double cy = intrinsic.at<double>(1, 2);

	for (int i = 0; i < pts_prev.size(); i++)
	{
		const double u1 = pts_prev[i].x;
		const double v1 = pts_prev[i].y;
		const double d1 = pts_prev[i].z;
		const double u2 = pts_next[i].x;
		const double v2 = pts_next[i].y;
		const double d2 = pts_next[i].z;

		double z1 = d1 / 1000;
		double x1 = z1*(u1 - cx) / fx;
		double y1 = z1*(v1 - cy) / fy;
		double z2 = d2 / 1000;
		double x2 = z2*(u2 - cx) / fx;
		double y2 = z2*(v2 - cy) / fy;

		prev[i].x = x1;
		prev[i].y = y1;
		prev[i].z = z1;
		next[i].x = x2;
		next[i].y = y2;
		next[i].z = z2;

	}
	cv::Mat A(N, 9, CV_64F);
	for (int i = 0; i < N; ++i)
	{

		const double x1 = prev[i].x;
		const double y1 = prev[i].y;
		const double z1 = prev[i].z;
		const double x2 = next[i].x;
		const double y2 = next[i].y;
		const double z2 = next[i].z;

		double* ai = A.ptr<double>(i);

		ai[0] = x2*x1;
		ai[1] = y2*x1;
		ai[2] = z2*x1;
		ai[3] = x2*y1;
		ai[4] = y2*y1;
		ai[5] = z2*y1;
		ai[6] = x2*z1;
		ai[7] = y2*z1;
		ai[8] = z2*z1;
	}
	cv::Mat u, w, vt;

	//cv::eigen(A.t()*A, w, vt);
	cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	cv::Mat Fpre = vt.row(8).reshape(0, 3);

	//This part is for testing.
	Mat result;
	Mat temp1 = Mat::zeros(1, 3, CV_64F);
	Mat temp2 = Mat::zeros(3, 1, CV_64F);
	for (int i = 0; i < N; i++)
	{
		temp1.at<double>(0, 0) = prev[i].x;
		temp1.at<double>(0, 1) = prev[i].y;
		temp1.at<double>(0, 2) = prev[i].z;
		temp2.at<double>(0, 0) = next[i].x;
		temp2.at<double>(1, 0) = next[i].y;
		temp2.at<double>(2, 0) = next[i].z;

		result = temp1*Fpre*temp2;
		cout << "This is test in RANSC: " << abs(result.at<double>(0, 0)) << endl;

	}

	cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

	w.at<double>(2) = 0;

	cv::Mat F_norm = u*cv::Mat::diag(w)*vt;

	return F_norm;
}
//Using Ransc to select best matches.(2D version)
vector<int> RANSC2D(vector<KeyPoint> keyPoint1, vector<KeyPoint> keyPoint2, vector<DMatch> goodMatchePoints, double treshold)
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
		fundemental = findF2D(selPoints1, selPoints2);

		Mat temp1 = Mat::zeros(1, 3, CV_64F);
		Mat temp2 = Mat::zeros(3, 1, CV_64F);
		Mat result;
		//cout << selPoints1[0].x << endl;
		vector<double> results;
		//cout << "Test Results:" << endl;
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
			//cout << tempresult << endl;
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
//Using Ransc to select best matches.(3D version)
vector<int> RANSC3D(vector<cv::Point3d> PointSet1, vector<cv::Point3d> PointSet2, double treshold, Mat intrinsic)
{
	vector<int> random;
	vector<int> select;
	int iternum = 0;
	cv::Mat fundemental;
	int M = 0;
	while (true)
	{
		iternum++;
		int tempM = 0;
		if (PointSet1.size() > 8)
			random = randvec(PointSet1.size());
		else
		{
			for (int i = 0; i < 8; i++)
				random.push_back(i);

		}

		vector<cv::Point3d> Set1(8);
		vector<cv::Point3d> Set2(8);
		for (int i = 0; i < 8; i++)
		{
			Set1[i].x = PointSet1[random[i]].x;
			Set1[i].y = PointSet1[random[i]].y;
			Set1[i].z = PointSet1[random[i]].z;
			Set2[i].x = PointSet2[random[i]].x;
			Set2[i].y = PointSet2[random[i]].y;
			Set2[i].z = PointSet2[random[i]].z;

		}

		// Compute F matrix from 8 matches
		fundemental = findF3D(Set1, Set2, intrinsic);

		std::vector<cv::Point3d> prev(PointSet1.size());
		std::vector<cv::Point3d> next(PointSet1.size());

		//cv::Mat T1, T2;
		//Normalize(pts_prev, pts_prev_norm, T1);
		//Normalize(pts_next, pts_next_norm, T2);
		double fx = intrinsic.at<double>(0, 0);
		double fy = intrinsic.at<double>(1, 1);
		double cx = intrinsic.at<double>(0, 2);
		double cy = intrinsic.at<double>(1, 2);

		for (int i = 0; i < PointSet1.size(); i++)
		{
			const double u1 = PointSet1[i].x;
			const double v1 = PointSet1[i].y;
			const double d1 = PointSet1[i].z;
			const double u2 = PointSet2[i].x;
			const double v2 = PointSet2[i].y;
			const double d2 = PointSet2[i].z;

			double z1 = d1 / 1000;
			double x1 = z1*(u1 - cx) / fx;
			double y1 = z1*(v1 - cy) / fy;
			double z2 = d2 / 1000;
			double x2 = z2*(u2 - cx) / fx;
			double y2 = z2*(v2 - cy) / fy;

			prev[i].x = x1;
			prev[i].y = y1;
			prev[i].z = z1;
			next[i].x = x2;
			next[i].y = y2;
			next[i].z = z2;

		}

		Mat temp1 = Mat::zeros(1, 3, CV_64F);
		Mat temp2 = Mat::zeros(3, 1, CV_64F);
		Mat result;
		//cout << selPoints1[0].x << endl;
		//cout << "Test Results:" << endl;
		for (int i = 0; i < prev.size(); i++)
		{
			temp1.at<double>(0, 0) = next[i].x;
			temp1.at<double>(0, 1) = next[i].y;
			temp1.at<double>(0, 2) = next[i].z;
			temp2.at<double>(0, 0) = prev[i].x;
			temp2.at<double>(1, 0) = prev[i].y;
			temp2.at<double>(2, 0) = prev[i].z;

			result = temp1*fundemental*temp2;
			double tempresult = abs(result.at<double>(0, 0));
			//cout << tempresult << endl;
			if (tempresult < treshold)
				tempM++;
		}
		if (tempM > M)
		{
			M = tempM;
			select = random;
		}
		if (iternum > 200)
		{
			cout << "Maching Numbers: " << M << " over " << PointSet1.size() << endl;
			return select;
			break;
		}
	}

}
//Fit data to solvePnP routine.
void SolveSp(vector<cv::Point3d> PointSet1, vector<cv::Point2f> selPoints2, Mat camera_matrix, Mat* Rotation, Mat* Transit, bool Ransc)
{
	cv::Mat rotation_vector; // Rotation in axis-angle form
	cv::Mat translation_vector;
	cv::Mat rotation;
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

	double fx = camera_matrix.at<double>(0, 0);
	double fy = camera_matrix.at<double>(1, 1);
	double cx = camera_matrix.at<double>(0, 2);
	double cy = camera_matrix.at<double>(1, 2);

	for (int i = 0; i < PointSet1.size(); i++)
	{
		PointSet1[i].z = PointSet1[i].z / 1000;
		PointSet1[i].x = PointSet1[i].z *(PointSet1[i].x - cx) / fx;
		PointSet1[i].y = PointSet1[i].z *(PointSet1[i].y - cy) / fy;

	}

	if (Ransc)
		cv::solvePnPRansac(PointSet1, selPoints2, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
	else
		cv::solvePnP(PointSet1, selPoints2, camera_matrix, dist_coeffs, rotation_vector, translation_vector, false, CV_EPNP);

	cv::Rodrigues(rotation_vector, *Rotation);
	rotation = *Rotation;
	cout << "Rotation Vector " << endl;
	cout << rotation.at<double>(0, 0) << " " << rotation.at<double>(0, 1) << " " << rotation.at<double>(0, 2) << endl;
	cout << rotation.at<double>(1, 0) << " " << rotation.at<double>(1, 1) << " " << rotation.at<double>(1, 2) << endl;
	cout << rotation.at<double>(2, 0) << " " << rotation.at<double>(2, 1) << " " << rotation.at<double>(2, 2) << endl;
	(*Transit) = translation_vector;
	cout << "Translation Vector" << endl;
	cout << translation_vector.at<double>(0, 0) << endl;
	cout << translation_vector.at<double>(1, 0) << endl;
	cout << translation_vector.at<double>(2, 0) << endl;

	//vector<cv::Point2d> projectedPoints;
	//cv::projectPoints(PointSet1, rotation_vector, translation_vector, camera_matrix, dist_coeffs, projectedPoints);

	//for (int i = 0; i < PointSet1.size(); i++)
	//{
	//	cout <<  "Points Compare: " << endl;
	//	cout << selPoints2[i].x << " " << selPoints2[i].y << endl;
	//	cout << projectedPoints[i].x << " " << projectedPoints[i].y << endl;
	//}

}

cv::Point3d findCentroids(vector<cv::Point3d> PointSet)
{
	double x_c = 0;
	double y_c = 0;
	double z_c = 0;
	int size = PointSet.size();
	for (int i = 0; i < size; i++)
	{
		x_c += PointSet[i].x;
		y_c += PointSet[i].y;
		z_c += PointSet[i].z;
	}
	return (Point3d(x_c / size, y_c / size, z_c / size));
}

void Registration(vector<cv::Point3d> PointSet1, vector<cv::Point3d> PointSet2, Mat intrinsic, Mat* Rotation, Mat* Transit)
{
	double fx = intrinsic.at<double>(0, 0);
	double fy = intrinsic.at<double>(1, 1);
	double cx = intrinsic.at<double>(0, 2);
	double cy = intrinsic.at<double>(1, 2);
	Mat R = Mat::zeros(3, 3, CV_64F);
	Mat T = Mat::zeros(3, 1, CV_64F);
	int N = PointSet1.size();

	for (int i = 0; i < N; i++)
	{
		PointSet1[i].z = PointSet1[i].z / 1000;
		PointSet1[i].x = PointSet1[i].z *(PointSet1[i].x - cx) / fx;
		PointSet1[i].y = PointSet1[i].z *(PointSet1[i].y - cy) / fy;

		PointSet2[i].z = PointSet2[i].z / 1000;
		PointSet2[i].x = PointSet2[i].z *(PointSet2[i].x - cx) / fx;
		PointSet2[i].y = PointSet2[i].z *(PointSet2[i].y - cy) / fy;
	}

	Point3d c1 = findCentroids(PointSet1);
	Point3d c2 = findCentroids(PointSet2);

	Mat H = Mat::zeros(3, 3, CV_64F);
	Mat temp1 = Mat::zeros(3, 1, CV_64F);
	Mat temp2 = Mat::zeros(1, 3, CV_64F);

	for (int i = 0; i < N; i++)
	{
		temp1.at<double>(0, 0) = PointSet1[i].x - c1.x;
		temp1.at<double>(1, 0) = PointSet1[i].y - c1.y;
		temp1.at<double>(2, 0) = PointSet1[i].z - c1.z;

		temp2.at<double>(0, 0) = PointSet2[i].x - c2.x;
		temp2.at<double>(0, 1) = PointSet2[i].y - c2.y;
		temp2.at<double>(0, 2) = PointSet2[i].z - c2.z;

		H = H + temp1*temp2;
	}
	Mat U, w, VT;
	cv::SVD fsvd(H);
	U = fsvd.u;
	w = fsvd.w;
	VT = fsvd.vt;

	R = VT.t()*U.t();
	cout << "Rotation Vector " << endl;
	cout << R.at<double>(0, 0) << " " << R.at<double>(0, 1) << " " << R.at<double>(0, 2) << endl;
	cout << R.at<double>(1, 0) << " " << R.at<double>(1, 1) << " " << R.at<double>(1, 2) << endl;
	cout << R.at<double>(2, 0) << " " << R.at<double>(2, 1) << " " << R.at<double>(2, 2) << endl;

	cout << "Check Point: |R|>0 ==> " << determinant(R) << endl;
	if (determinant(R) < 0)
	{
		R.col(2) = -R.col(2);
	}

	Mat c_1 = Mat::zeros(3, 1, CV_64F);
	Mat c_2 = Mat::zeros(3, 1, CV_64F);

	c_1.at<double>(0, 0) = c1.x;
	c_1.at<double>(1, 0) = c1.y;
	c_1.at<double>(2, 0) = c1.z;

	c_2.at<double>(0, 0) = c2.x;
	c_2.at<double>(1, 0) = c2.y;
	c_2.at<double>(2, 0) = c2.z;

	T = -R*c_1 + c_2;

	cout << "Translation Vector" << endl;
	cout << T.at<double>(0, 0) << endl;
	cout << T.at<double>(1, 0) << endl;
	cout << T.at<double>(2, 0) << endl;

	(*Rotation) = R;
	(*Transit) = T;
}

void test(Mat image1, Mat image2)
{
	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keyPoint1;
	std::vector<cv::KeyPoint> keyPoint2;
	detector.detect(image1, keyPoint1);
	detector.detect(image2, keyPoint2);
	drawKeypoints(image1, keyPoint1, image1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(image2, keyPoint2, image2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("KeyPoints of image1", image1);
	cv::imshow("KeyPoints of image2", image2);
	Mat imageDesc1, imageDesc2;
	detector.compute(image1, keyPoint1, imageDesc1);
	detector.compute(image2, keyPoint2, imageDesc2);

}