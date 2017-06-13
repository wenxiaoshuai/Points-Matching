#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>  

using namespace cv;
using namespace std;

int main()
{
	Mat image01 = imread("img01.jpg");
	Mat image02 = imread("img02.jpg");
	Mat image1, image2;
	image1 = image01.clone();
	image2 = image02.clone();

	//提取特征点    
	SurfFeatureDetector surfDetector(8000);  //hessianThreshold,海塞矩阵阈值，并不是限定特征点的个数   
	vector<KeyPoint> keyPoint1, keyPoint2;
	surfDetector.detect(image1, keyPoint1);
	surfDetector.detect(image2, keyPoint2);

	//绘制特征点    
	drawKeypoints(image1, keyPoint1, image1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(image2, keyPoint2, image2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("KeyPoints of image1", image1);
	imshow("KeyPoints of image2", image2);


	//特征点描述，为下边的特征点匹配做准备    
	SurfDescriptorExtractor SurfDescriptor;
	Mat imageDesc1, imageDesc2;
	SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
	SurfDescriptor.compute(image2, keyPoint2, imageDesc2);

	//特征点匹配并显示匹配结果    
	//BruteForceMatcher<L2<float>> matcher;    
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());

	//提取强特征点  
	double minMatch = 1;
	double maxMatch = 0;
	for (int i = 0; i<matchePoints.size(); i++)
	{
		//匹配值最大最小值获取  
		minMatch = minMatch>matchePoints[i].distance ? matchePoints[i].distance : minMatch;
		maxMatch = maxMatch<matchePoints[i].distance ? matchePoints[i].distance : maxMatch;
	}
	//最大最小值输出  
	cout << "The Best Match is： " << minMatch << endl;
	cout << "The Worst Match is： " << maxMatch << endl;

	//获取排在前边的几个最优匹配结果  
	vector<DMatch> goodMatchePoints;
	for (int i = 0; i<matchePoints.size(); i++)
	{
		if (matchePoints[i].distance<minMatch + (maxMatch - minMatch) / 2)
		{
			goodMatchePoints.push_back(matchePoints[i]);
		}
	}

	//绘制最优匹配点  
	Mat imageOutput;
	drawMatches(image01, keyPoint1, image02, keyPoint2, goodMatchePoints, imageOutput, Scalar::all(-1),
		Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("Mathch Points", imageOutput);
	waitKey();
	return 0;
}

