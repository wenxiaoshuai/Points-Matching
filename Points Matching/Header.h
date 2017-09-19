//Header for Opencv
#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"  
//Header for File Stream.
#include <fstream>
//Header for Math
#include "math.h"
#include <iostream>

using namespace cv;
using namespace std;

//Selecting good matches based on double match and Surf descriptor distance.
void SelectMatching(vector<DMatch> matchPoints, vector<DMatch> matchPoints2, vector<DMatch>* goodMatchePoints, vector<DMatch>* usingMatchePoints);

// A filter to remove poor matches.
void FilterMatching(vector<KeyPoint> keyPoint1, vector<KeyPoint> keyPoint2, vector<DMatch>* MatchePoints);

//function for RANSC.
vector<int> randvec(int Num);

//Using Ransc to select best matches.(2D version)
vector<int> RANSC2D(vector<KeyPoint> keyPoint1, vector<KeyPoint> keyPoint2, vector<DMatch> goodMatchePoints, double treshold);

//Using Ransc to select best matches.(3D version)
vector<int> RANSC3D(vector<cv::Point3d> PointSet1, vector<cv::Point3d> PointSet2, double treshold, Mat intrinsic);

//2D based 8 points algorithm.
cv::Mat findF2D(const std::vector<cv::Point2f> pts_prev, const std::vector<cv::Point2f> pts_next);

//3D based 8 points algorithm.
cv::Mat findF3D(const std::vector<cv::Point3d> pts_prev, const std::vector<cv::Point3d> pts_next, Mat intrinsic);
cv::Mat findF3DF(const std::vector<cv::Point3d> pts_prev, const std::vector<cv::Point3d> pts_next, Mat intrinsic);

//Decompose R and T from Essential Matrix.
void SolveRt(Mat Essential, Mat* Rotation1, Mat* Rotation2, Mat* Transit);
void function(Mat M, Mat R, Mat T, Mat* temp1, Mat* temp2);

//Chose R&T with 2D points.(technically, the T here is a*t.(t is the real transit vector)).
int ChooseRT1(Mat R1, Mat R2, Mat T, Mat intrinsic, vector<cv::Point3d> Point1, vector<cv::Point3d> Point2);

//Calculate the distance between two vector.
double getDis(Mat a, int n);

//Get the Maximum distance within the vector sets.
double getMaxDis(Mat a, int size);

//Using 3D points set to calculate the real R and t.
void ChooseRT2(Mat R1, Mat R2, Mat* T, Mat intrinsic, vector<cv::Point3d> Point1, vector<cv::Point3d> Point2);

//Normalizing the data.
void Normalize(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_norm, cv::Mat& T);

//Fit data to solvePnP routine.
void SolveSp(vector<cv::Point3d> PointSet1, vector<cv::Point2f> selPoints2, Mat camera_matrix, Mat* Rotation, Mat* Transit, bool Ransc = false);

//Validating the results from SolvePnp routine.
void ValidatePnp(Mat R, Mat t, Mat intrinsic, vector<cv::Point3d> Point1, vector<cv::Point3d> Point2);

//Using Registartion to calculate the R and t.
cv::Point3d findCentroids(vector<cv::Point3d> PointSet);
void Registration(vector<cv::Point3d> PointSet1, vector<cv::Point3d> PointSet2, Mat intrinsic, Mat* Rotation, Mat* Transit);

//Write 3D points sets to file in order to porcessing the GO-ICP Algorithm.
void tofile(Mat intrinsic, vector<cv::Point3d> Point1, vector<cv::Point3d> Point2);

void test(Mat image1, Mat image2);