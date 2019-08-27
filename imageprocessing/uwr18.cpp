#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/bgsegm.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/algorithm/copy.hpp>
,
using namespace cv;
using namespace std;

#define H_MAX 180
#define H_MIN 160
#define S_MAX 255
#define S_MIN 50
#define V_MAX 255
#define V_MIN 50

Ptr<BackgroundSubtractor> pMOG;
//コンパイル: g++ uwr18.cpp -I/usr/local/include/opencv2 -I/usr/local/include/opencv -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_bgsegm -lopencv_videoio

int main(int argc, const char **argv)
{
    const int CAP_WIDTH = 640;
    const int CAP_HEIGHT = 480;
    VideoCapture capl(1); //デバイスのオープン
    VideoCapture capr(2); //デバイスのオープン
    capl.set(CV_CAP_PROP_FRAME_WIDTH, CAP_WIDTH);
    capl.set(CV_CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT);
    capr.set(CV_CAP_PROP_FRAME_WIDTH, CAP_WIDTH);
    capr.set(CV_CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT);

    if (!capl.isOpened()) //カメラデバイスが正常にオープンしたか確認．
    {
        printf("camera l not found\n");
        //return -1;
    }
    if (!capr.isOpened()) //カメラデバイスが正常にオープンしたか確認．
    {
        printf("camera r not found\n");
        //return -1;
    }

    Mat framel; //取得したフレーム
    Mat framer; //取得したフレーム
    Mat image_blurred_with_5x5_kernel;
    capl.read(framel);
    imwrite("imgl.jpg", framel);
    //Mat imgl = imread("dots3.jpg", IMREAD_COLOR);//imgl
    //Mat imgr = imread("dots3.jpg", IMREAD_COLOR);//imgr
    Mat imgl = imread("imgl.jpg", IMREAD_COLOR); //imgl
    Mat imgr = imread("imgl.jpg", IMREAD_COLOR); //imgr
    printf("img prepared\n");
    Mat hsvimg, hsvfilteredimgl, hsvfilteredimgr;
    Scalar s_min = Scalar(H_MIN, S_MIN, V_MIN);
    Scalar s_max = Scalar(H_MAX, S_MAX, V_MAX);

    while(true){
        //capl.read(framel);
        //imwrite("imgl.jpg", framel);
        //capr.read(framer);
        //imwrite("imgr.jpg", framel);
        //imgl = imread("imgl.jpg", IMREAD_COLOR);
        //imgr = imread("imgr.jpg", IMREAD_COLOR);
        GaussianBlur(imgl, image_blurred_with_5x5_kernel, Size(5, 5), 0);
        GaussianBlur(imgr, image_blurred_with_5x5_kernel, Size(5, 5), 0);
        cvtColor(imgl, hsvimg, CV_BGR2HSV, 3);
        inRange(hsvimg, s_min, s_max, hsvfilteredimgl);
        cvtColor(imgr, hsvimg, CV_BGR2HSV, 3);
        inRange(hsvimg, s_min, s_max, hsvfilteredimgr);

        //namedWindow("hsvfiltered", WINDOW_AUTOSIZE);
        //imshow("hsvfiltered", hsvfilteredimgl);

        int whitepixell = 0;
        int whitepixelr = 0;
        for (int y = 0; y < imgl.rows; y++)
        {
            for (int x = 0; x < imgl.cols; x++)
            {
                if (hsvfilteredimgl.at<unsigned char>(y, x) > 200)
                {
                    whitepixell++;
                }
                if (hsvfilteredimgr.at<unsigned char>(y, x) > 200)
                {
                    whitepixelr++;
                }
            }
        }
        printf("whitepixell = %d, whitepixelr = %d\n", whitepixell, whitepixelr);
        Mat matl(whitepixell, 2, CV_32F);
        Mat matr(whitepixelr, 2, CV_32F);
        whitepixell = 0;
        whitepixelr = 0;
        for (int y = 0; y < imgl.rows; y++)
        {
            for (int x = 0; x < imgl.cols; x++)
            {

                if (hsvfilteredimgl.at<unsigned char>(y, x) > 200)
                {
                    matl.at<float>(whitepixell, 0) = x;
                    matl.at<float>(whitepixell, 1) = y;
                    //samples.at<float>(whitepixel, 2) = 0;
                    whitepixell++;
                }
                if (hsvfilteredimgr.at<unsigned char>(y, x) > 200)
                {
                    matr.at<float>(whitepixelr, 0) = x;
                    matr.at<float>(whitepixelr, 1) = y;
                    //samples.at<float>(whitepixel, 2) = 0;
                    whitepixelr++;
                }
            }
        }
        printf("whitepixel2 = %d\n", whitepixell);

        Mat labelsl;
        Mat labelsr;
        int attempts = 1;
        Mat centersl(2, 2, matl.type());
        Mat centersr(2, 2, matr.type());
        TermCriteria criteria{TermCriteria::COUNT, 1, 100};
        kmeans(matl, 2, labelsl, criteria, attempts, KMEANS_PP_CENTERS, centersl);
        kmeans(matr, 2, labelsr, criteria, attempts, KMEANS_PP_CENTERS, centersr);
        float f = 0.0;
        float T = 0.0;
        float xl = 0.0, xr = 0.0, yl = 0.0, yr = 0.0;
        float Zp = f * T / (xl - xr);
        float Xp = xl * T / (xl - xr) - 1 / 2 * T;
        float Yp = yl * T / (xl - xr);

        Mat new_image(imgl.size(), imgl.type());
        int cluster_idx;
        int number = 0;
        int color = 1;
        int K = 2;
        for (int y = 0; y < imgl.rows; y++)
        {
            for (int x = 0; x < imgl.cols; x++)
            {
                if (hsvfilteredimgr.at<unsigned char>(y, x) > 200)
                {
                    cluster_idx = labelsr.at<int>(number, 0);
                    new_image.at<Vec3b>(y, x)[0] = (cluster_idx + 2) * 20;  //centers.at<float>(cluster_idx, 0) * color;
                    new_image.at<Vec3b>(y, x)[1] = cluster_idx * 200;       //centers.at<float>(cluster_idx, 1) * color;
                    new_image.at<Vec3b>(y, x)[2] = (K - cluster_idx) * 200; //centers.at<float>(cluster_idx, 2) * color;
                    number++;
                }
                else
                {
                    new_image.at<Vec3b>(y, x)[0] = 0;
                    new_image.at<Vec3b>(y, x)[1] = 0;
                    new_image.at<Vec3b>(y, x)[2] = 0;
                }
            }
        }
        std::cout << "centersl: " << centersl << endl;
        std::cout << "centersr: " << centersr << endl;
        //namedWindow("clustered_image", WINDOW_AUTOSIZE);
        //imshow("clustered_image", new_image);
        //waitKey(0);
        //destroyAllWindows();
        break;
    }
    return 0;
}