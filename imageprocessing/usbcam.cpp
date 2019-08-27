#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/opencv_lib.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/format.hpp>
using namespace cv;

//compile   g++ usbcam.cpp -I/usr/local/include/opencv2 -I/usr/local/include/opencv -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_bgsegm -lopencv_videoio

int main(int argh, char *argv[])
{
    const int CAP_WIDTH = 640;
    const int CAP_HEIGHT = 480;
    VideoCapture cap(1); //デバイスのオープン
    //cap.open(0);//こっちでも良い．
    cap.set(CV_CAP_PROP_FRAME_WIDTH, CAP_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT);

    if (!cap.isOpened()) //カメラデバイスが正常にオープンしたか確認．
    {
        //読み込みに失敗したときの処理
        return -1;
    }

    Mat frame;          //取得したフレーム
    //while (cap.read(frame)) //無限ループ
    {
        //
        //取得したフレーム画像に対して，クレースケール変換や2値化などの処理を書き込む．
        //
        cap.read(frame);
        //imshow("win", frame); //画像を表示．
        imwrite("img.jpg", frame);
        //const int key = waitKey(1);
        //if (key == 'q' /*113*/) //qボタンが押されたとき
        //{
        //    break; //whileループから抜ける．
        //}
        //else if (key == 's' /*115*/) //sが押されたとき
        //{
        //    //フレーム画像を保存する．
        //    imwrite("img.jpg", frame);
        //}
    }
    destroyAllWindows();
    return 0;
}