#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat srcImage = imread("lena.jpg");      //加载图像文件
	namedWindow("lena", WINDOW_AUTOSIZE);   //设置显示图像的窗口标题为lena,属性为自动调整大小
	imshow("lena", srcImage);               //显示图片

	waitKey(0);

	return 0;
}