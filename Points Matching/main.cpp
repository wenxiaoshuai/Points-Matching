
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
	SelectMatching(matchePoints, matchePoints2, &goodMatchePoints, &UsingMatches);
	//Filtering the matching pairs.
	//FilterMatching(keyPoint1, keyPoint2, &goodMatchePoints);

	vector<int> usepointIndexes1;
	vector<int> usepointIndexes2;
	vector<int> pointIndexes1;
	vector<int> pointIndexes2;
	vector<int> RANSCIndexes1;
	vector<int> RANSCIndexes2;

	cout << "Good Matches are:" << endl;
	for (int i = 0; i < goodMatchePoints.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, goodMatchePoints[i].queryIdx, goodMatchePoints[i].trainIdx);
	}
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
	intrinsic.at<double>(0, 0) = 1082.628;
	intrinsic.at<double>(1, 1) = 1082.628;
	intrinsic.at<double>(2, 2) = 1;
	intrinsic.at<double>(0, 2) = 960.125;
	intrinsic.at<double>(1, 2) = 539.314;
	transpose(intrinsic, intrinsicT);

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
			if (depth1 > 8000)
				depth1 = 0;
			//cout << (int)depth01.at<Vec3b>(y1, x1)[2] << "   " << (int)depth01.at<Vec3b>(y1, x1)[1] << "   " << (int)depth01.at<Vec3b>(y1, x1)[0] << endl;
		}
		if ((int)depth02.at<uchar>(y2, x2) != 255)
		{
			depth2 = (int)depth02.at<Vec3b>(y2, x2)[2] * 1000 + (int)depth02.at<Vec3b>(y2, x2)[1] * 100 + (int)depth02.at<Vec3b>(y2, x2)[0];
			if (depth2 > 8000)
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
		double d1 = (int)depth01.at<Vec3b>(v1, u1)[2] * 1000 + (int)depth01.at<Vec3b>(v1, u1)[1] * 100 + (int)depth01.at<Vec3b>(v1, u1)[0];
		PointSet1[i].z = d1;
		double u2 = selPoints2[i].x;
		double v2 = selPoints2[i].y;
		PointSet2[i].x = selPoints2[i].x;
		PointSet2[i].y = selPoints2[i].y;
		double d2 = (int)depth02.at<Vec3b>(v2, u2)[2] * 1000 + (int)depth02.at<Vec3b>(v2, u2)[1] * 100 + (int)depth02.at<Vec3b>(v2, u2)[0];
		PointSet2[i].z = d2;
	}

	//Using 8-Points Algorithm to obtain Fundamental Matrix.
	vector<int> select = RANSC3D(PointSet1, PointSet2, 0.001, intrinsic);

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

	//Construct a fake matching points.
	vector<cv::Point2f> fakePoints1, fakePoints2;
	for (int i = 0; i < RANSCPoints1.size(); i++)
	{
		fakePoints1.push_back(RANSCPoints1[i]);
		fakePoints2.push_back(Point(RANSCPoints1[i].x + 200, RANSCPoints1[i].y));
	}

	for (int i = 0; i < RANSCPoints1.size(); i++)
	{
		double u1 = RANSCPoints1[i].x;
		double v1 = RANSCPoints1[i].y;
		RanscSet1[i].x = RANSCPoints1[i].x;
		RanscSet1[i].y = RANSCPoints1[i].y;
		double d1 = (int)depth01.at<Vec3b>(v1, u1)[2] * 1000 + (int)depth01.at<Vec3b>(v1, u1)[1] * 100 + (int)depth01.at<Vec3b>(v1, u1)[0];
		RanscSet1[i].z = d1;
		cout << "RANSC Set1: " << RanscSet1[i].x << " " << RanscSet1[i].y << " " << RanscSet1[i].z << endl;
		double u2 = RANSCPoints2[i].x;
		double v2 = RANSCPoints2[i].y;
		RanscSet2[i].x = RANSCPoints1[i].x+200;
		RanscSet2[i].y = RANSCPoints1[i].y;
		double d2 = (int)depth02.at<Vec3b>(v2, u2)[2] * 1000 + (int)depth02.at<Vec3b>(v2, u2)[1] * 100 + (int)depth02.at<Vec3b>(v2, u2)[0];
		RanscSet2[i].z = d1;
		cout << "RANSC Set2: " << RanscSet2[i].x << " " << RanscSet2[i].y << " " << RanscSet2[i].z << endl;
		cout << "---------------------------------" << endl;
	}

	// Compute F matrix from 8 matches
	fundemental = cv::findFundamentalMat(
		cv::Mat(fakePoints1), // points in first image
		cv::Mat(fakePoints2), // points in second image
		CV_FM_8POINT); // 8-point method

	////Now we get the Fundamental Matrix F.
	//cv::Mat F = Mat::zeros(3, 3, CV_64F);
	//double value = 0;
	//F = findF3D(RANSCPoints1, RANSCPoints2, &value);

	////Correct Matches.
	////cv::correctMatches(fundemental, usePoints1, usePoints2, newpoint1, newpoint2);

	//cout << "This is F:" << endl;
	//cout << F.at<double>(0, 0) << " " << F.at<double>(0, 1) << F.at<double>(0, 2) << endl;
	//cout << F.at<double>(1, 0) << " " << F.at<double>(1, 1) << F.at<double>(1, 2) << endl;
	//cout << F.at<double>(2, 0) << " " << F.at<double>(2, 1) << F.at<double>(2, 2) << endl;

	//Mat temp1 = Mat::zeros(1, 3, CV_64F);
	//Mat temp2 = Mat::zeros(3, 1, CV_64F);
	//Mat result;
	////cout << selPoints1[0].x << endl;
	//vector<double> results;
	//for (int i = 0; i < selPoints1.size(); i++)
	//{
	//	temp1.at<double>(0, 0) = selPoints1[i].x;
	//	temp1.at<double>(0, 1) = selPoints1[i].y;
	//	temp1.at<double>(0, 2) = 1;
	//	temp2.at<double>(0, 0) = selPoints2[i].x;
	//	temp2.at<double>(1, 0) = selPoints2[i].y;
	//	temp2.at<double>(2, 0) = 1;

	//	result = temp1*F*temp2;
	//	cout << "$$$" << abs(result.at<double>(0, 0)) << endl;
	//}

	//Mat temp11 = Mat::zeros(1, 3, CV_64F);
	//Mat temp22 = Mat::zeros(3, 1, CV_64F);
	//Mat result0;
	////cout << selPoints1[0].x << endl;
	//for (int i = 0; i < selPoints2.size(); i++)
	//{
	//	temp11.at<double>(0, 0) = selPoints1[i].x;
	//	temp11.at<double>(0, 1) = selPoints1[i].y;
	//	temp11.at<double>(0, 2) = 1;
	//	temp22.at<double>(0, 0) = selPoints2[i].x;
	//	temp22.at<double>(1, 0) = selPoints2[i].y;
	//	temp22.at<double>(2, 0) = 1;

	//	result0 = temp11*fundemental*temp22;
	//	cout << "@@@" << abs(result0.at<double>(0, 0)) << endl;
	//}


	//find Essential Matrix.
	EssentialMatrix = intrinsicT*fundemental*intrinsic;

	cout << "Essential Matrix" << endl;
	cout << EssentialMatrix.at<double>(0, 0) << " " << EssentialMatrix.at<double>(0, 1) << " " << EssentialMatrix.at<double>(0, 2) << endl;
	cout << EssentialMatrix.at<double>(1, 0) << " " << EssentialMatrix.at<double>(1, 1) << " " << EssentialMatrix.at<double>(1, 2) << endl;
	cout << EssentialMatrix.at<double>(2, 0) << " " << EssentialMatrix.at<double>(2, 1) << " " << EssentialMatrix.at<double>(2, 2) << endl;


	//1.Sove the R & T from Essential.
	Mat Rotation1 = Mat::zeros(3, 3, CV_64F);
	Mat Rotation2 = Mat::zeros(3, 3, CV_64F);
	Mat Transit = Mat::zeros(3, 1, CV_64F);
	Mat PnP_R1 = Mat::zeros(3, 3, CV_64F);
	Mat PnP_T1 = Mat::zeros(3, 1, CV_64F);
	Mat PnP_R2 = Mat::zeros(3, 3, CV_64F);
	Mat PnP_T2 = Mat::zeros(3, 1, CV_64F);
	SolveRt(EssentialMatrix, &Rotation1, &Rotation2, &Transit);

	//Correct Matches.
	cv::correctMatches(fundemental, selPoints1, selPoints2, newpoint1, newpoint2);

	//2.Solve R & T with SolvePnP.
	int choose = -1;
	SolveSp(RanscSet1, fakePoints2, intrinsic, &PnP_R1, &PnP_T1, true);
	SolveSp(RanscSet2, fakePoints1, intrinsic, &PnP_R2, &PnP_T2, true);

	choose = ChooseRT2(PnP_R1, Rotation1, PnP_T2, intrinsic, RanscSet1, RanscSet2);



	//Write the 3D points to a file for ICP Alg.
	//tofile(intrinsic, PointSet1, PointSet2);


	//Mat H1 = Mat::zeros(3, 3, CV_64F);
	//Mat H2 = Mat::zeros(3, 1, CV_64F);
	//if (choose == -1)
	//	function(intrinsic, Rotation0, Transit0, &H1, &H2);
	//if (choose == 0)
	//	function(intrinsic, Rotation1, Transit, &H1, &H2);
	//if (choose == 1)
	//	function(intrinsic, Rotation1, -Transit, &H1, &H2);
	//if (choose == 2)
	//	function(intrinsic, Rotation2, Transit, &H1, &H2);
	//if (choose == 3)
	//	function(intrinsic, Rotation2, -Transit, &H1, &H2);

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
	Mat imageOutput;//UsingMatches
	cv::drawMatches(image01, keyPoint1, image02, keyPoint2, Matches, imageOutput, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//cv::imshow("Mathch Points", imageOutput);

	// Draw Epipolar Lines.
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
