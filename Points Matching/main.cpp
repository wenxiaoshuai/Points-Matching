
#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>  
#include "Header.h"



int main()
{
	//Read images.
	Mat image01 = imread("ColorImage1.jpg");
	Mat image02 = imread("ColorImage2.jpg");
	//Mat image01 = imread("img1.bmp");
	//Mat image02 = imread("img2.bmp");

	FileStorage fs1("DepthData1.xml", FileStorage::READ);
	Mat depth01;
	fs1["depth"] >> depth01;
	FileStorage fs2("DepthData2.xml", FileStorage::READ);
	Mat depth02;
	fs2["depth"] >> depth02;
	//	cout << (int)depth02.at<uchar>(1418, 975) << endl;
	Mat image1, image2;
	Mat img1, img2;
	image1 = image01.clone();
	image2 = image02.clone();
	img1 = image01.clone();
	img2 = image02.clone();

	//Extracting SURF feature.    
	SurfFeatureDetector surfDetector(5000);  //Set hessianThreshold  
	vector<KeyPoint> keyPoint1, keyPoint2;
	surfDetector.detect(image1, keyPoint1);
	surfDetector.detect(image2, keyPoint2);

	//Plot the features.
	drawKeypoints(image1, keyPoint1, image1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(image2, keyPoint2, image2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img2, keyPoint2, img2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//cv::imshow("KeyPoints of image1", image1);
	//cv::imshow("KeyPoints of image2", image2);


	//Obtain the descriptors of the feature.
	SurfDescriptorExtractor SurfDescriptor;
	Mat imageDesc1, imageDesc2;
	SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
	SurfDescriptor.compute(image2, keyPoint2, imageDesc2);

	//Matching the features.  
	FlannBasedMatcher matcher;
	//BruteForceMatcher<L2<double>> matcher;
	vector<DMatch> matchePoints, matchePoints2;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	matcher.match(imageDesc2, imageDesc1, matchePoints2, Mat());
	vector<DMatch> goodMatchePoints;
	vector<DMatch> UsingMatches;
	vector<DMatch> validMatchePoints;
	vector<DMatch> test;
	SelectMatching(matchePoints, matchePoints2, &goodMatchePoints, &UsingMatches);

	//Filtering the matching pairs.
	//FilterMatching(keyPoint1, keyPoint2, &goodMatchePoints);

	vector<int> usepointIndexes1;
	vector<int> usepointIndexes2;
	vector<int> pointIndexes1;
	vector<int> pointIndexes2;
	vector<int> RANSCIndexes1;
	vector<int> RANSCIndexes2;

	vector<cv::Point2f> selPoints1, selPoints2, newpoint1, newpoint2;
	vector<cv::Point2f> usePoints1, usePoints2;
	vector<cv::Point2f> RANSCPoints1, RANSCPoints2;
	cv::Mat fundemental;
	Mat EssentialMatrix;

	//Define Intrinsic Matrix
	Mat intrinsic = Mat::zeros(3, 3, CV_64F);
	Mat intrinsicT = Mat::zeros(3, 3, CV_64F);
	//depth camera intrinsic.
	//intrinsic.at<double>(0, 0) = 365.299;
	//intrinsic.at<double>(1, 1) = 365.299;
	//intrinsic.at<double>(2, 2) = 1;
	//intrinsic.at<double>(0, 2) = 256.398;
	//intrinsic.at<double>(1, 2) = 206.882;
	//estimated color camera intrinsic.
	intrinsic.at<double>(0, 0) = 1082.628;//1033.174
	intrinsic.at<double>(1, 1) = 1082.628;//1032.664
	intrinsic.at<double>(2, 2) = 1;
	intrinsic.at<double>(0, 2) = 960.125;//972.342
	intrinsic.at<double>(1, 2) = 539.314;//532.647
	transpose(intrinsic, intrinsicT);


	for (int i = 0; i < goodMatchePoints.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, goodMatchePoints[i].queryIdx, goodMatchePoints[i].trainIdx);
		int x1 = keyPoint1[goodMatchePoints[i].queryIdx].pt.x;
		int y1 = keyPoint1[goodMatchePoints[i].queryIdx].pt.y;
		int x2 = keyPoint2[goodMatchePoints[i].trainIdx].pt.x;
		int y2 = keyPoint2[goodMatchePoints[i].trainIdx].pt.y;
		int depth1 = 0;
		int depth2 = 0;
		if (depth01.at<double>(y1, x1) != -1)
		{
			depth1 = depth01.at<double>(y1, x1);
			if (depth1 > 5000)
				depth1 = 0;
		}
		if (depth02.at<double>(y2, x2) != -1)
		{
			depth2 = depth02.at<double>(y2, x2);
			if (depth2 > 5000)
				depth2 = 0;
		}
		if (depth1 > 500 && depth2 > 500)
		{
			validMatchePoints.push_back(goodMatchePoints[i]);
		}

	}

	for (int i = 0; i < UsingMatches.size(); i++)
	{
		usepointIndexes1.push_back(UsingMatches[i].queryIdx);
		usepointIndexes2.push_back(UsingMatches[i].trainIdx);
	}

	for (int i = 0; i < validMatchePoints.size(); i++)
	{
		pointIndexes1.push_back(validMatchePoints[i].queryIdx);
		pointIndexes2.push_back(validMatchePoints[i].trainIdx);
	}

	// Convert keypoints into Point2f
	KeyPoint::convert(keyPoint1, selPoints1, pointIndexes1);
	KeyPoint::convert(keyPoint2, selPoints2, pointIndexes2);
	KeyPoint::convert(keyPoint1, usePoints1, usepointIndexes1);
	KeyPoint::convert(keyPoint2, usePoints2, usepointIndexes2);

	//Construct a valid 3D point set.
	vector<cv::Point3d> PointSet1(selPoints1.size());
	vector<cv::Point3d> PointSet2(selPoints2.size());

	for (int i = 0; i < selPoints1.size(); i++)
	{
		double u1 = selPoints1[i].x;
		double v1 = selPoints1[i].y;
		PointSet1[i].x = selPoints1[i].x;
		PointSet1[i].y = selPoints1[i].y;
		double d1 = depth01.at<double>(v1, u1);
		PointSet1[i].z = d1;
		double u2 = selPoints2[i].x;
		double v2 = selPoints2[i].y;
		PointSet2[i].x = selPoints2[i].x;
		PointSet2[i].y = selPoints2[i].y;
		double d2 = depth02.at<double>(v2, u2);
		PointSet2[i].z = d2;
	}

	//Using 8-Points Algorithm to obtain Fundamental Matrix.
	vector<int> select = RANSC3D(PointSet1, PointSet2, 0.0001, intrinsic);
	//vector<int> select = RANSC2D(keyPoint1, keyPoint2, validMatchePoints, 0.001);

	vector<DMatch> Matches;
	for (int i = 0; i < 8; i++)
	{
		Matches.push_back(validMatchePoints[select[i]]);
	}

	for (int i = 0; i < Matches.size(); i++)
	{
		RANSCIndexes1.push_back(Matches[i].queryIdx);
		RANSCIndexes2.push_back(Matches[i].trainIdx);
	}

	// Convert keypoints into Point2f
	KeyPoint::convert(keyPoint1, RANSCPoints1, RANSCIndexes1);
	KeyPoint::convert(keyPoint2, RANSCPoints2, RANSCIndexes2);


	//Construct a ransac 3D point set.
	vector<cv::Point3d> RanscSet1(RANSCPoints1.size());
	vector<cv::Point3d> RanscSet2(RANSCPoints2.size());

	for (int i = 0; i < RANSCPoints1.size(); i++)
	{
		double u1 = RANSCPoints1[i].x;
		double v1 = RANSCPoints1[i].y;
		RanscSet1[i].x = RANSCPoints1[i].x;
		RanscSet1[i].y = RANSCPoints1[i].y;
		double d1 = depth01.at<double>(v1, u1);
		RanscSet1[i].z = d1;
		cout << "RANSC Set1: " << RanscSet1[i].x << " " << RanscSet1[i].y << " " << RanscSet1[i].z << endl;
		double u2 = RANSCPoints2[i].x;
		double v2 = RANSCPoints2[i].y;
		RanscSet2[i].x = RANSCPoints2[i].x;
		RanscSet2[i].y = RANSCPoints2[i].y;
		double d2 = depth02.at<double>(v2, u2);
		RanscSet2[i].z = d2;
		cout << "RANSC Set2: " << RanscSet2[i].x << " " << RanscSet2[i].y << " " << RanscSet2[i].z << endl;
		cout << "---------------------------------" << endl;
	}

	// Compute F matrix from 8 matches
	//1. 2D 8-points Algorithm.
	fundemental = cv::findFundamentalMat(
		cv::Mat(RANSCPoints1), // points in first image
		cv::Mat(RANSCPoints2), // points in second image
		CV_FM_8POINT); // 8-point method
	//2. 3D 8-points Algorithm.
	cv::Mat F = Mat::zeros(3, 3, CV_64F);
	//F = findF3DF(RanscSet2, RanscSet1, intrinsic);
	////Now we get the Fundamental Matrix F.

	//find Essential Matrix.
	EssentialMatrix = intrinsicT*fundemental*intrinsic;

	cout << "Essential Matrix" << endl;
	cout << EssentialMatrix.at<double>(0, 0) << " " << EssentialMatrix.at<double>(0, 1) << " " << EssentialMatrix.at<double>(0, 2) << endl;
	cout << EssentialMatrix.at<double>(1, 0) << " " << EssentialMatrix.at<double>(1, 1) << " " << EssentialMatrix.at<double>(1, 2) << endl;
	cout << EssentialMatrix.at<double>(2, 0) << " " << EssentialMatrix.at<double>(2, 1) << " " << EssentialMatrix.at<double>(2, 2) << endl;

	//cout << "F Matrix" << endl;
	//cout << F.at<double>(0, 0) << " " << F.at<double>(0, 1) << " " << F.at<double>(0, 2) << endl;
	//cout << F.at<double>(1, 0) << " " << F.at<double>(1, 1) << " " << F.at<double>(1, 2) << endl;
	//cout << F.at<double>(2, 0) << " " << F.at<double>(2, 1) << " " << F.at<double>(2, 2) << endl;


	//1.Solve the R & T from Essential.
	Mat Rotation1 = Mat::zeros(3, 3, CV_64F);
	Mat Rotation2 = Mat::zeros(3, 3, CV_64F);
	Mat Transit = Mat::zeros(3, 1, CV_64F);
	SolveRt(EssentialMatrix, &Rotation1, &Rotation2, &Transit);
	ChooseRT2(Rotation1, Rotation2, &Transit, intrinsic, RanscSet1, RanscSet2);
	ValidatePnp(Rotation2, Transit, intrinsic, RanscSet1, RanscSet2);
	//2.Solve R & T with SolvePnP.
	Mat PnP_R = Mat::zeros(3, 3, CV_64F);
	Mat PnP_T = Mat::zeros(3, 1, CV_64F);
	Mat Reg_R = Mat::zeros(3, 3, CV_64F);
	Mat Reg_T = Mat::zeros(3, 1, CV_64F);
	SolveSp(RanscSet1, RANSCPoints2, intrinsic, &PnP_R, &PnP_T, false);
	ValidatePnp(PnP_R, PnP_T, intrinsic, RanscSet1, RanscSet2);

	//3. Solve R & T from 3D Registration.
	Registration(RanscSet1, RanscSet2, intrinsic, &Reg_R, &Reg_T);
	ValidatePnp(Reg_R, Reg_T, intrinsic, RanscSet1, RanscSet2);

	//Write the 3D points to a file for ICP Alg.
	//tofile(intrinsic, PointSet1, PointSet2);


	//Mat H1 = Mat::zeros(3, 3, CV_64F);
	//Mat H2 = Mat::zeros(3, 1, CV_64F);
	//	function(intrinsic, Rotation0, Transit0, &H1, &H2);
	//	function(intrinsic, Rotation1, Transit, &H1, &H2);
	;

	//Mat showimg(1080, 1920, CV_8UC3, Scalar(255, 255, 255));
	//cv::imshow("Show Image", showimg);
	//for (int j = 0; j < depth01.rows; j++)
	//	for (int i = 0; i < depth01.cols; i++)
	//	{
	//		double d = (int)depth01.at<Vec3b>(j, i)[2] * 1000 + (int)depth01.at<Vec3b>(j, i)[1] * 100 + (int)depth01.at<Vec3b>(j, i)[0];
	//		if (d > 500 && d < 5000)
	//		{
	//			double z = d / 1000;
	//			Mat p1 = Mat::zeros(3, 1, CV_64F);
	//			Mat p2 = Mat::zeros(3, 1, CV_64F);
	//			p1.at<double>(0, 0) = i;
	//			p1.at<double>(1, 0) = j;
	//			p1.at<double>(2, 0) = 1;
	//			p2 = z*H1*p1 - H2;
	//			double u1 = p2.at<double>(0, 0) / p2.at<double>(2, 0);
	//			double v1 = p2.at<double>(1, 0) / p2.at<double>(2, 0);
	//			//cout << endl;
	//			//cout << "(u,v)"<<u1 << " " << v1 << endl;
	//			//cout << "(i,j)" << i << " " << j << endl;
	//			if (u1 >= 0 && u1 < showimg.cols && v1 >= 0 && v1 < showimg.rows)
	//			{

	//				showimg.at<Vec3b>(v1, u1)[0] = image01.at<Vec3b>(j, i)[0];
	//				showimg.at<Vec3b>(v1, u1)[1] = image01.at<Vec3b>(j, i)[1];
	//				showimg.at<Vec3b>(v1, u1)[2] = image01.at<Vec3b>(j, i)[2];
	//				//cout << image01.at<Vec3b>(j, i)[0] << " " << image01.at<Vec3b>(j, i)[1] << " " << image01.at<Vec3b>(j, i)[2] << endl;
	//			}

	//		}
	//	}
	//cv::imshow("Show Image", showimg);
	//imwrite("show_PnPRansc.jpg", showimg);

	//Draw Good Matching Points.

	//Correct Matches.
	cv::correctMatches(F, RANSCPoints1, RANSCPoints2, newpoint1, newpoint2);
	Mat imageOutput;//UsingMatches
	cv::drawMatches(image01, keyPoint1, image02, keyPoint2, Matches, imageOutput, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::imshow("Mathch Points", imageOutput);

	// Draw Epipolar Lines.
	std::vector<cv::Vec3f> lines1;
	cv::computeCorrespondEpilines(
		cv::Mat(newpoint1), // image points
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

	cv::imshow("Image Epilines in img2", img2);
	cv::waitKey();
	return 0;
}
