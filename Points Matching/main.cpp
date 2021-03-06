﻿#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>  

using namespace cv;
using namespace std;

int main()
{
	//Read images.
	//Mat image01 = imread("img01.jpg");
	//Mat image02 = imread("img02.jpg");
	Mat image01 = imread("img1.bmp");
	Mat image02 = imread("img2.bmp");
	Mat image1, image2;
	Mat img1, img2;
	image1 = image01.clone();
	image2 = image02.clone();
	img1 = image01.clone();
	img2 = image02.clone();
	//Extracting SURF feature.    
	SurfFeatureDetector surfDetector(8000);  //Set hessianThreshold  
	vector<KeyPoint> keyPoint1, keyPoint2;
	surfDetector.detect(image1, keyPoint1);
	surfDetector.detect(image2, keyPoint2);

	//Plot the features.
	drawKeypoints(image1, keyPoint1, image1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(image2, keyPoint2, image2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img2, keyPoint2, img2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("KeyPoints of image1", image1);
	imshow("KeyPoints of image2", image2);


	//Obtain the descriptors of the feature.
	SurfDescriptorExtractor SurfDescriptor;
	Mat imageDesc1, imageDesc2;
	SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
	SurfDescriptor.compute(image2, keyPoint2, imageDesc2);

	//Matching the features.  
	//BruteForceMatcher<L2<float>> matcher;    
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());

	//Selecting strong features.
	double minMatch = 1;
	double maxMatch = 0;
	for (int i = 0; i < matchePoints.size(); i++)
	{
		//get the max and min value of all the matches.
		minMatch = minMatch > matchePoints[i].distance ? matchePoints[i].distance : minMatch;
		maxMatch = maxMatch < matchePoints[i].distance ? matchePoints[i].distance : maxMatch;
	}

	cout << "The Best Match is： " << minMatch << endl;
	cout << "The Worst Match is： " << maxMatch << endl;

	//Get the top matching points.
	vector<DMatch> goodMatchePoints;
	for (int i = 0; i < matchePoints.size(); i++)
	{
		if (matchePoints[i].distance < minMatch + (maxMatch - minMatch) / 2)
		{
			goodMatchePoints.push_back(matchePoints[i]);
		}
	}

	vector<int> pointIndexes1;
	vector<int> pointIndexes2;
	cout << "Good Matches are:" << endl;
	for (int i = 0; i < goodMatchePoints.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, goodMatchePoints[i].queryIdx, goodMatchePoints[i].trainIdx);
		pointIndexes1.push_back(goodMatchePoints[i].queryIdx);
		pointIndexes2.push_back(goodMatchePoints[i].trainIdx);
	}
	//Draw Good Matching Points 
	Mat imageOutput;
	drawMatches(image01, keyPoint1, image02, keyPoint2, goodMatchePoints, imageOutput, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Mathch Points", imageOutput);

	//Using 8-Points Algorithm to obtain Fundamental Matrix.

	// Convert keypoints into Point2f
	vector<cv::Point2f> selPoints1, selPoints2;
	KeyPoint::convert(keyPoint1, selPoints1, pointIndexes1);
	KeyPoint::convert(keyPoint2, selPoints2, pointIndexes2);


	// Compute F matrix from 7 matches
	cv::Mat fundemental = cv::findFundamentalMat(
		cv::Mat(selPoints1), // points in first image
		cv::Mat(selPoints2), // points in second image
		CV_FM_7POINT); // 7-point method


	//Now we get the Fundamental Matrix F.

	Mat temp1 = Mat::zeros(1, 3, CV_64F);
	Mat temp2 = Mat::zeros(3, 1, CV_64F);
	Mat result;
	double sum = 0;
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

		cout << "result = " << i << " " << result.at<double>(0, 0) << endl;
		sum += abs(result.at<double>(0, 0));
	}

	cout << "The average value is  " << sum / goodMatchePoints.size() << endl;
	

	// draw the left points corresponding epipolar lines in right image
	std::vector<cv::Vec3f> lines1;
	cv::computeCorrespondEpilines(
		cv::Mat(selPoints1), // image points
		1, // in image 1 (can also be 2)
		fundemental, // F matrix
		lines1); // vector of epipolar lines
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
	imshow("Image Epilines", img2);
	
	waitKey();
	return 0;
}