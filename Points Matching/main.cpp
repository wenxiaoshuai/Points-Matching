﻿
#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>  

using namespace cv;
using namespace std;

void SelectMatching(vector<DMatch> matchPoints, vector<DMatch> matchPoints2, vector<DMatch>* goodMatchePoints)
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

	vector<DMatch> tempMatchePoints;
	for (int i = 0; i < matchPoints.size(); i++)
	{
		if (matchPoints[i].distance < minMatch + (maxMatch - minMatch) / 3)
		{
			tempMatchePoints.push_back(matchPoints[i]);
		}
	}

	for (int i = 0; i < tempMatchePoints.size(); i++)
	{
		int match = tempMatchePoints[i].trainIdx;
		if (matchPoints2[match].trainIdx == tempMatchePoints[i].queryIdx)
		{
			(*goodMatchePoints).push_back(tempMatchePoints[i]);
		}
	}
}
void FilterMatching(vector<KeyPoint> keyPoint1, vector<KeyPoint> keyPoint2, vector<DMatch>* MatchePoints)
{
	//Get rid of the noise matches.
	double slopesum = 0;
	double avgslope = 0;
	int num = (*MatchePoints).size();
	for (int i = 0; i < (*MatchePoints).size(); i++)
	{
		int temp1 = (*MatchePoints)[i].queryIdx;
		int temp2 = (*MatchePoints)[i].trainIdx;
		Point point1 = keyPoint1[temp1].pt;
		Point point2 = keyPoint2[temp2].pt;
		double tempslope = (double)(point2.y - point1.y) / (double)(point2.x + 1080 - point1.x);//1080 here is the width of the image.Change it when applied in Kinect.
		slopesum += tempslope;
	}
	avgslope = slopesum / num;
	int nn = 0;
	for (int i = 0; i < (*MatchePoints).size(); i++)
	{
		int temp1 = (*MatchePoints)[i].queryIdx;
		int temp2 = (*MatchePoints)[i].trainIdx;
		Point point1 = keyPoint1[temp1].pt;
		Point point2 = keyPoint2[temp2].pt;
		double tempslope = (double)(point2.y - point1.y) / (double)(point2.x + 1080 - point1.x);
		cout << avgslope + abs(avgslope) << " " << avgslope - abs(avgslope) << endl;
		if (tempslope < avgslope + 2*abs(avgslope) && tempslope > avgslope - 2*abs(avgslope))
		{
			(*MatchePoints).erase((*MatchePoints).begin() + i - nn);
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


int main()
{
	//Read images.
	Mat image01 = imread("ColorImage1.jpg");
	Mat image02 = imread("ColorImage2.jpg");
	//Mat image01 = imread("img1.bmp");
	//Mat image02 = imread("img2.bmp");
	Mat depth01 = imread("DepthImage1.jpg");
	Mat depth02 = imread("DepthImage2.jpg");
	//	cout << (int)depth02.at<uchar>(1418, 975) << endl;
	Mat image1, image2;
	Mat img1, img2;
	image1 = image01.clone();
	image2 = image02.clone();
	img1 = image01.clone();
	img2 = image02.clone();
	//Extracting SURF feature.    
	SurfFeatureDetector surfDetector(6000);  //Set hessianThreshold  
	vector<KeyPoint> keyPoint1, keyPoint2;
	surfDetector.detect(image1, keyPoint1);
	surfDetector.detect(image2, keyPoint2);

	//Plot the features.
	drawKeypoints(image1, keyPoint1, image1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(image2, keyPoint2, image2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img2, keyPoint2, img2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	cv::imshow("KeyPoints of image1", image1);
	cv::imshow("KeyPoints of image2", image2);


	//Obtain the descriptors of the feature.
	SurfDescriptorExtractor SurfDescriptor;
	Mat imageDesc1, imageDesc2;
	SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
	SurfDescriptor.compute(image2, keyPoint2, imageDesc2);

	//Matching the features.  
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints, matchePoints2;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	matcher.match(imageDesc2, imageDesc1, matchePoints2, Mat());
	vector<DMatch> goodMatchePoints;
	SelectMatching(matchePoints, matchePoints2, &goodMatchePoints);
	//Filtering the matching pairs.
	FilterMatching(keyPoint1, keyPoint2, &goodMatchePoints);

	vector<int> pointIndexes1;
	vector<int> pointIndexes2;
	cout << "Good Matches are:" << endl;

	vector<cv::Point2f> selPoints1, selPoints2, newpoint1, newpoint2;
	cv::Mat fundemental;
	Mat EssentialMatrix;
	int iternum = 0;
	double threshold = 100;//At first, I wanna use a iteration to update the essential matrix.
	//	while (true)
	{
		//iternum++;
		//cout << "-----------------------------------------------------" << endl;
		double sum = 0;

		pointIndexes1.clear();
		pointIndexes2.clear();
		int num = goodMatchePoints.size();
		cout << "size is: " << num << endl;
		int nums = 0;
		/*
		//Test Matches.
		goodMatchePoints.erase(goodMatchePoints.begin());
		goodMatchePoints.erase(goodMatchePoints.begin() + 1);
		goodMatchePoints.erase(goodMatchePoints.begin());
		goodMatchePoints.erase(goodMatchePoints.begin() + 2);
		goodMatchePoints.erase(goodMatchePoints.begin() + 2);
		goodMatchePoints.erase(goodMatchePoints.begin() + 4);
		goodMatchePoints.erase(goodMatchePoints.begin() + 4);
		goodMatchePoints.erase(goodMatchePoints.begin() + 6);
		goodMatchePoints.erase(goodMatchePoints.begin() + 8);
		goodMatchePoints.erase(goodMatchePoints.begin() + 12);
		goodMatchePoints.erase(goodMatchePoints.begin() + 5);
		//消灭3个映射到1个
		goodMatchePoints.erase(goodMatchePoints.begin() + 8);
		goodMatchePoints.erase(goodMatchePoints.begin() + 11);
		int n = goodMatchePoints.size();
		for (int i = 12; i < n; i++)
		{
			goodMatchePoints.pop_back();
		}
		*/

		for (int i = 0; i < goodMatchePoints.size(); i++)
		{
			printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, goodMatchePoints[i].queryIdx, goodMatchePoints[i].trainIdx);
			int x1 = keyPoint1[goodMatchePoints[i].queryIdx].pt.x;
			int y1 = keyPoint1[goodMatchePoints[i].queryIdx].pt.y;
			int x2 = keyPoint2[goodMatchePoints[i].trainIdx].pt.x;
			int y2 = keyPoint2[goodMatchePoints[i].trainIdx].pt.y;
			int depth1 = 0;
			int depth2 = 0;
			if ((int)depth01.at<uchar>(y1, x1) != 255)
			{
				depth1 = (int)depth01.at<Vec3b>(y1, x1)[2] * 1000 + (int)depth01.at<Vec3b>(y1, x1)[1] * 100 + (int)depth01.at<Vec3b>(y1, x1)[0];
				if (depth1 > 5000)
					depth1 = 0;
				//cout << (int)depth01.at<Vec3b>(y1, x1)[2] << "   " << (int)depth01.at<Vec3b>(y1, x1)[1] << "   " << (int)depth01.at<Vec3b>(y1, x1)[0] << endl;
			}
			if ((int)depth02.at<uchar>(y2, x2) != 255)
			{
				depth2 = (int)depth02.at<Vec3b>(y2, x2)[2] * 1000 + (int)depth02.at<Vec3b>(y2, x2)[1] * 100 + (int)depth02.at<Vec3b>(y2, x2)[0];
				if (depth2 > 5000)
					depth2 = 0;
			}
			if (depth1 > 500 && depth2 > 500)
			{
				nums++;
				cout << "Num " << nums << "    depth1 = " << depth1 << "  depth2= " << depth2 << endl;
				pointIndexes1.push_back(goodMatchePoints[i].queryIdx);
				pointIndexes2.push_back(goodMatchePoints[i].trainIdx);
			}

		}



		//Using 8-Points Algorithm to obtain Fundamental Matrix.

		// Convert keypoints into Point2f

		KeyPoint::convert(keyPoint1, selPoints1, pointIndexes1);
		KeyPoint::convert(keyPoint2, selPoints2, pointIndexes2);


		// Compute F matrix from 7 matches
		fundemental = cv::findFundamentalMat(
			cv::Mat(selPoints1), // points in first image
			cv::Mat(selPoints2), // points in second image
			CV_FM_8POINT); // 7-point method

		//Now we get the Fundamental Matrix F.

		Mat temp1 = Mat::zeros(1, 3, CV_64F);
		Mat temp2 = Mat::zeros(3, 1, CV_64F);
		Mat result;
		//cout << selPoints1[0].x << endl;



		for (int i = 0; i < goodMatchePoints.size(); i++)
		{
			temp1.at<double>(0, 0) = keyPoint1[goodMatchePoints[i].queryIdx].pt.x;
			temp1.at<double>(0, 1) = keyPoint1[goodMatchePoints[i].queryIdx].pt.y;
			temp1.at<double>(0, 2) = 1;
			temp2.at<double>(0, 0) = keyPoint2[goodMatchePoints[i].trainIdx].pt.x;
			temp2.at<double>(1, 0) = keyPoint2[goodMatchePoints[i].trainIdx].pt.y;
			temp2.at<double>(2, 0) = 1;

			result = temp1*fundemental*temp2;

			//cout << "result = " << i << " " << result.at<double>(0, 0) << endl;
			sum += abs(result.at<double>(0, 0));
			//if (result.at<double>(0, 0) > threshold)
			//{
			//	goodMatchePoints.erase(goodMatchePoints.begin() + i);
			//}
		}


		cout << "The average value is  " << sum / num << endl;
		//cout << "iter num is: " << iternum << endl;
		//if (iternum > 20 || threshold == sum / num || goodMatchePoints.size() < 8)
		//	break;
		//threshold = sum / num;

		//find Essential Matrix.
		Mat intrinsic = Mat::zeros(3, 3, CV_64F);
		Mat intrinsicT = Mat::zeros(3, 3, CV_64F);
		//depth camera intrinsic.
		//intrinsic.at<double>(0, 0) = 365.299;
		//intrinsic.at<double>(1, 1) = 365.299;
		//intrinsic.at<double>(2, 2) = 1;
		//intrinsic.at<double>(0, 2) = 256.398;
		//intrinsic.at<double>(1, 2) = 206.882;
		//estimated color camera intrinsic.
		intrinsic.at<double>(0, 0) = 1082.628;
		intrinsic.at<double>(1, 1) = 1082.628;
		intrinsic.at<double>(2, 2) = 1;
		intrinsic.at<double>(0, 2) = 960.125;
		intrinsic.at<double>(1, 2) = 539.314;
		transpose(intrinsic, intrinsicT);
		EssentialMatrix = intrinsicT*fundemental*intrinsic;

		cout << "Essential Matrix" << endl;
		cout << EssentialMatrix.at<double>(0, 0) << " " << EssentialMatrix.at<double>(0, 1) << " " << EssentialMatrix.at<double>(0, 2) << endl;
		cout << EssentialMatrix.at<double>(1, 0) << " " << EssentialMatrix.at<double>(1, 1) << " " << EssentialMatrix.at<double>(1, 2) << endl;
		cout << EssentialMatrix.at<double>(2, 0) << " " << EssentialMatrix.at<double>(2, 1) << " " << EssentialMatrix.at<double>(2, 2) << endl;

		//Sove the R & T from Essential.
		Mat Rotation1 = Mat::zeros(3, 3, CV_64F);
		Mat Rotation2 = Mat::zeros(3, 3, CV_64F);
		Mat Transit = Mat::zeros(3, 1, CV_64F);
		SolveRt(EssentialMatrix, &Rotation1, &Rotation2, &Transit);
		//Get Coeffcient Matrices.
		Mat H1 = Mat::zeros(3, 3, CV_64F);
		Mat H2 = Mat::zeros(3, 1, CV_64F);
		function(intrinsic, Rotation1, -Transit, &H1, &H2);

		/*
		Mat showimg = image02.clone();
		for (int j = 0; j < depth01.rows; j++)
			for (int i = 0; i < depth01.cols; i++)
			{
				double d = (int)depth01.at<Vec3b>(j, i)[2] * 1000 + (int)depth01.at<Vec3b>(j, i)[1] * 100 + (int)depth01.at<Vec3b>(j, i)[0];
				if (d > 500 && d < 8000)
				{
					double z = d / 1000;
					Mat p1 = Mat::zeros(3, 1, CV_64F);
					Mat p2 = Mat::zeros(3, 1, CV_64F);
					p1.at<double>(0, 0) = i;
					p1.at<double>(1, 0) = j;
					p1.at<double>(2, 0) = 1;
					p2 = z*H1*p1 - H2;
					double u1 = p2.at<double>(0, 0) / p2.at<double>(2, 0);
					double v1 = p2.at<double>(1, 0) / p2.at<double>(2, 0);
					//cout << endl;
					//cout << "(u,v)"<<u1 << " " << v1 << endl;
					//cout << "(i,j)" << i << " " << j << endl;
					if (u1 >= 0 && u1 < showimg.cols&&v1 >= 0 && v1 < showimg.rows)
					{
						showimg.at<Vec3b>(v1, u1)[0] = image01.at<Vec3b>(j, i)[0];
						showimg.at<Vec3b>(v1, u1)[1] = image01.at<Vec3b>(j, i)[1];
						showimg.at<Vec3b>(v1, u1)[2] = image01.at<Vec3b>(j, i)[2];
						//cout << image01.at<Vec3b>(j, i)[0] << " " << image01.at<Vec3b>(j, i)[1] << " " << image01.at<Vec3b>(j, i)[2] << endl;
					}

				}
			}
		cv::imshow("Show Image", showimg);
		*/

		//Validate the results.
		cv::correctMatches(fundemental, selPoints1, selPoints2, newpoint1, newpoint2);
		for (int i = 0; i < newpoint1.size(); i++)
		{
			double u = selPoints1[i].x;
			double v = selPoints1[i].y;
			double d = (int)depth01.at<Vec3b>(v, u)[2] * 1000 + (int)depth01.at<Vec3b>(v, u)[1] * 100 + (int)depth01.at<Vec3b>(v, u)[0];
			if (d > 500 && d < 4500)
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
				cout << "newpoint2 " << selPoints2[i].x << " " << selPoints2[i].y << endl;
				cout << "(u1,v1)  " << u1 << " " << v1 << endl;
				cout << "-----------------------------------" << endl;
			}

		}


	}




	//Draw Good Matching Points.
	Mat imageOutput;
	cv::drawMatches(image01, keyPoint1, image02, keyPoint2, goodMatchePoints, imageOutput, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imshow("Mathch Points", imageOutput);

	// Draw Epipolar Lines.
	cv::correctMatches(fundemental, selPoints1, selPoints2, newpoint1, newpoint2);

	std::vector<cv::Vec3f> lines1;
	std::vector<cv::Vec3f> lines2;
	cv::computeCorrespondEpilines(
		cv::Mat(newpoint1), // image points
		1, // in image 1 (can also be 2)
		fundemental, // F matrix
		lines1); // vector of epipolar lines
				 // for all epipolar lines
	cv::computeCorrespondEpilines(
		cv::Mat(newpoint2), // image points
		2, // in image 1 (can also be 2)
		fundemental, // F matrix
		lines2); // vector of epipolar lines
				 // for all epipolar lines
	for (vector<cv::Vec3f>::const_iterator it = lines1.begin();
		it != lines1.end(); ++it) {
		// draw the line between first and last column
		cv::line(img2,
			cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(img2.cols, -((*it)[2] +
			(*it)[0] * img2.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}
	for (vector<cv::Vec3f>::const_iterator it = lines2.begin();
		it != lines2.end(); ++it) {
		// draw the line between first and last column
		cv::line(img1,
			cv::Point(0, -(*it)[2] / (*it)[1]),
			cv::Point(img1.cols, -((*it)[2] +
			(*it)[0] * img1.cols) / (*it)[1]),
			cv::Scalar(255, 255, 255));
	}
	//cv::imshow("Image Epilines in img1", img1);
	//cv::imshow("Image Epilines in img2", img2);

	cv::waitKey();
	return 0;
}
