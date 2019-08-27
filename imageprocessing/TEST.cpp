#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/opencv.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/format.hpp>
using namespace cv;
using namespace std;
#define Pi 3.1415

Ptr<BackgroundSubtractor> pMOG; //MOG Background subtractor
//コンパイル: g++ test.cpp -I/usr/local/include/opencv2 -I/usr/local/include/opencv -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_bgsegm

float variancefunc(Mat &mat, int Rn, int K2, float xu = 0.0, float yu = 0.0)
{
     float variance = 0.0;
     if (xu == 0.0 && yu == 0.0)
     {
          for (int i = 0; i < Rn; i++)
          {
               xu += mat.at<float>(i, 0);
               yu += mat.at<float>(i, 1);
          }
          xu = xu / Rn;
          yu = yu / Rn;
     }
     std::cout << "  xu: " << xu << endl;
     std::cout << "  yu: " << yu << endl;
     for (int i = 0; i < Rn; i++)
     {
          variance += pow(mat.at<float>(i, 0) - xu, 2) + pow(mat.at<float>(i, 1) - yu, 2);
          if (i % 100 == 0)
          {
               //std::cout << "variance: " << variance << endl;
               //std::cout << "variance: " << mat.at<float>(i, 0) << endl;
               //std::cout << "variance: " << pow((mat.at<float>(i, 0) + mat.at<float>(i, 1) - xu - yu), 2) << endl;
          }
     }
     return variance /= (Rn - K2);
}

float ID1func(Mat &mat, int Rn, float variance, float xu, float yu)
{
     float ID = 0.0;
     float xsub = 0.0;
     for (int i = 0; i < Rn, i++)
     {
          xsubu += pow(mat.at<float>(i, 0) - xu, 2) + pow(mat.at<float>(i, 1) - yu, 2);
     }
     return -log(pow(2 * Pi, 2) * pow(variance, 2)) - xsubu / (2 * variance) + log(Rn1 * Rn2 / pow(Rn1 + Rn2, 2));
}

float ID2func(Mat &mat1, Mat &mat2, int Rn1, int Rn2, float variance, float xu1, float yu1, float xu2, float yu2)
{
     float ID = 0.0;
     float xsubu = 0.0;
     for (int i = 0; i < Rn1; i++)
     {
          xsubu += pow(mat1.at<float>(i, 0) - xu1, 2) + pow(mat1.at<float>(i, 1) - yu1, 2);
     }
     for (int i = 0; i < Rn2; i++)
     {
          xsubu += pow(mat2.at<float>(i, 0) - xu2, 2) + pow(mat2.at<float>(i, 1) - yu2, 2);
     }
     return -log(pow(2 * Pi, 2) * pow(variance, 2)) - xsubu / (2 * variance) + log(Rn1 * Rn2 / pow(Rn1 + Rn2, 2));
}

float ID(Mat &mat, int R, int Rn, int K, int K2, float xu = 0.0, float yu = 0.0) //matのIDを求める
{
     float variance = 0.0;
     if (xu == 0.0 && yu == 0.0)
     {
          for (int i = 0; i < Rn; i++)
          {
               xu += mat.at<float>(i, 0);
               yu += mat.at<float>(i, 1);
          }
          xu = xu / Rn;
          yu = yu / Rn;
     }
     std::cout << "  xu: " << xu << endl;
     std::cout << "  yu: " << yu << endl;
     for (int i = 0; i < Rn; i++)
     {
          variance += pow(mat.at<float>(i, 0) - xu, 2) + pow(mat.at<float>(i, 1) - yu, 2);
          if (i % 100 == 0)
          {
               //std::cout << "variance: " << variance << endl;
               //std::cout << "variance: " << mat.at<float>(i, 0) << endl;
               //std::cout << "variance: " << pow((mat.at<float>(i, 0) + mat.at<float>(i, 1) - xu - yu), 2) << endl;
          }
     }
     variance /= (Rn - K2);
     printf("  variance = %f\n", variance);
     int p = 2;
     float ID = 0.0;
     float root2pi = pow(2 * Pi, 0.5);
     float tmp = 0;
     int d = 2 * p;
     //for (int i = 0; i < Rn; i++)
     {
          //ID += log(Rn / (root2pi * variance * R)) - (pow(mat.at<float>(i, 0) - xu, 2) + pow(mat.at<float>(i, 1) - yu, 2)) / (2 * variance);
          //tmp = log(Rn / (root2pi * variance * R)) - (pow(mat.at<float>(i, 0) - xu, 2) + pow(mat.at<float>(i, 1) - yu, 2)) / (2 * variance);
          ID = log(Rn) - Rn * log(R) - Rn * d / 2 * log(2 * Pi) - Rn / 2 * log(variance) - (Rn - K2) / 2;

          //if (i % 100 == 0)
          {
               //printf(" IDfactor = %f\n", tmp);
          }
     }
     //printf("K = %d, ID = %f\n", K, ID);
     return ID;
}

int xmeans2(Mat &mat){
     int Rn = mat.rows;
     float variance = variance()
}

int xmeans(Mat &mat, int *K, int R, float BICparent = 0.0)
{
     //std::cout << "rows: " << mat.rows << endl;
     //std::cout << "cols: " << mat.cols << endl;
     int Rn = mat.rows;
     printf("Rn = %d\n", Rn);
     int p = 2;
     int q = 2 * p;
     if (BICparent == 0.0)
     {
          float IDparent = ID(mat, R, Rn, *K, 1);
          BICparent = IDparent - q / 2 * log(Rn);
     }
     printf("K = %d, BICparent = %f, p/2*logR = %f\n", *K, BICparent, p / 2 * log(R));

     Mat labels;
     int attempts = 1;
     Mat centers(Rn, 1, mat.type());
     TermCriteria criteria{TermCriteria::COUNT, 1, 100};
     kmeans(mat, 2, labels, criteria, attempts, KMEANS_PP_CENTERS, centers);

     //std::cout << "size: " << centers.size() << endl;
     //std::cout << "centers: " << centers << endl;
     //std::cout << "labels: " << labels.size() << endl;

     float xuchild1 = 0.0;
     float yuchild1 = 0.0;
     float xuchild2 = 0.0;
     float yuchild2 = 0.0;
     int Rchild1 = 0;
     int Rchild2 = 0;
     for (int i = 0; i < Rn; i++)
     {
          if (labels.at<int>(i) == 0)
          {
               xuchild1 += mat.at<float>(i, 0);
               yuchild1 += mat.at<float>(i, 1);
               Rchild1++;
          }
          else
          {
               xuchild2 += mat.at<float>(i, 0);
               yuchild2 += mat.at<float>(i, 1);
               Rchild2++;
          }
     }
     xuchild1 = xuchild1 / Rchild1;
     yuchild1 = yuchild1 / Rchild1;
     xuchild2 = xuchild2 / Rchild2;
     yuchild2 = yuchild2 / Rchild2;
     //std::cout << "Rchild1: " << Rchild1 << endl;
     //std::cout << "Rchild2: " << Rchild2 << endl;
     //std::cout << "xuchild1: " << xuchild1 << endl;
     //std::cout << "yuchild1: " << yuchild1 << endl;
     //std::cout << "xuchild2: " << xuchild2 << endl;
     //std::cout << "yuchild2: " << yuchild2 << endl;
     Mat matchild1(Rchild1, 2, mat.type());
     Mat matchild2(Rchild2, 2, mat.type());
     int temp1 = 0;
     int temp2 = 0;
     for (int i = 0; i < Rn; i++)
     {
          if (labels.at<int>(i) == 0)
          {
               matchild1.at<float>(temp1, 0) = mat.at<float>(i, 0);
               matchild1.at<float>(temp1, 1) = mat.at<float>(i, 1);
               if (i % 100 == 0)
               {
                    //std::cout << "mati0: " << mat.at<float>(i, 0) << endl;
                    //std::cout << "mati1: " << mat.at<float>(i, 1) << endl;
               }
               temp1++;
          }
          if (labels.at<int>(i) == 1)
          {
               matchild2.at<float>(temp2, 0) = mat.at<float>(i, 0);
               matchild2.at<float>(temp2, 1) = mat.at<float>(i, 1);
               temp2++;
          }
     }

     float IDchild1 = ID(matchild1, Rn, Rchild1, *K, 2, xuchild1, yuchild1);
     float IDchild2 = ID(matchild2, Rn, Rchild2, *K, 2, xuchild2, yuchild2);
     float BICchild = IDchild1 + IDchild2 - q / 2 * log(Rn);
     //printf("BICchild1 = %f, BIChcild2 = %f, SUM = %f\n", BICchild1, BICchild2, BICchild1 + BICchild2);
     printf("BICchild = %f, IDchild1 = %f, IDchild2 = %f\n", BICchild, IDchild1, IDchild2);
     if (BICparent < BICchild && (*K) < 10)
     {
          (*K) = (*K) + 1;
          printf("BIC changed %d\n", *K);
          xmeans(matchild1, K, Rchild1, IDchild1 - q / 2 * log(Rn));
          xmeans(matchild2, K, Rchild2, IDchild2 - q / 2 * log(Rn));
     }
     else
     {
          printf("BIC Not changed\n");
     }
     //Mat matchild1(, 2, CV_32F);
     //Mat matchild2(, 2, CV_32F);
     return (*K);
}

int main(int argc, const char **argv)
{
     //Mat wallimg = imread("wall.jpeg", IMREAD_COLOR);
     //Mat wallimg = imread("pool2.jpg", IMREAD_COLOR);
     Mat wallimg = imread("whitebase.jpg", IMREAD_COLOR);
     Mat image_blurred_with_5x5_kernel;
     GaussianBlur(wallimg, image_blurred_with_5x5_kernel, Size(5, 5), 0);
     //Mat handimg = imread("hand.jpeg", IMREAD_COLOR);
     //Mat handimg = imread("pool_ball.jpg", IMREAD_COLOR);
     Mat handimg = imread("dots3.jpg", IMREAD_COLOR);
     GaussianBlur(handimg, image_blurred_with_5x5_kernel, Size(5, 5), 0);
     //namedWindow("wall", WINDOW_AUTOSIZE);
     //imshow("wall", wallimg);
     namedWindow("hand", WINDOW_AUTOSIZE);
     imshow("hand", handimg);

     Mat fgMaskMOG;                                  //fg mask generated by MOG method
     pMOG = bgsegm::createBackgroundSubtractorMOG(); //MOG approach
     pMOG->apply(wallimg, fgMaskMOG);
     pMOG->apply(handimg, fgMaskMOG);
     namedWindow("masked_hand", WINDOW_AUTOSIZE);
     imshow("masked_hand", fgMaskMOG);

     //fgMaskMOG = wallimg;
     //printf("rows = %d, cols = %d\n", fgMaskMOG.rows, fgMaskMOG.cols);
     //printf("%d %d %d\n", fgMaskMOG.at<Vec3b>(0, 280)[0], fgMaskMOG.at<Vec3b>(0, 280)[1], fgMaskMOG.at<Vec3b>(0, 280)[2]);
     int whitepixel = 0;
     for (int y = 0; y < fgMaskMOG.rows; y++)
     {
          for (int x = 0; x < fgMaskMOG.cols; x++)
          {
               if (fgMaskMOG.at<unsigned char>(y, x) > 200)
               {
                    whitepixel++;
               }
          }
     }
     Mat samples(whitepixel, 2, CV_32F);
     //Mat samples = fgMaskMOG;
     whitepixel = 0;
     for (int y = 0; y < fgMaskMOG.rows; y++)
     {
          for (int x = 0; x < fgMaskMOG.cols; x++)
          {

               if (fgMaskMOG.at<unsigned char>(y, x) > 200)
               {
                    samples.at<float>(whitepixel, 0) = x;
                    samples.at<float>(whitepixel, 1) = y;
                    //samples.at<float>(whitepixel, 2) = 0;
                    whitepixel++;
               }
          }
     }
     printf("whitepixel = %d\n", whitepixel);
     int K = 1;
     K = xmeans(samples, &K, whitepixel);
     printf("result K = %d\n", K);
     //std::cout << "size: " << centers.size() << endl;
     //std::cout << "centers: " << centers << endl;
     //std::cout << "labels: " << labels.size() << endl;

     waitKey(0);
     destroyAllWindows();

     return 0;
}
