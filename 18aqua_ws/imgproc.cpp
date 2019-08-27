#include <ros/ros.h>
#include <std_msgs/String.h>
#include <iostream>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv/cv.h>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/format.hpp>
using namespace cv;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    const int CAP_WIDTH = 640;
    const int CAP_HEIGHT = 480;
    VideoCapture cap1(1);
    //VideoCapture cap2(2);
    //cap1.set(CV_CAP_PROP_FRAME_WIDTH, CAP_WIDTH);
    //cap1.set(CV_CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT);
    
    //cap2.set(CV_CAP_PROP_FRAME_WIDTH, CAP_WIDTH);
    //cap2.set(CV_CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT);
    if (!cap1.isOpened()) //カメラデバイスが正常にオープンしたか確認．
    {
        //読み込みに失敗したときの処理
        printf("cam failed\n");
        return -1;
    }
    printf("cam ok\n");

    Mat frame;
    {
        cap1.read(frame);
        imwrite("img.jpg", frame);
        printf("jpg file maked\n");
    }
    destroyAllWindows();
    return 0;
}
