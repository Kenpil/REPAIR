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

double variancefunc(Mat &mat, int R, int K2, double *xu, double *yu)
{
    int Rn = mat.rows;
    double variance = 0.0;
    if (*xu == 0.0 && *yu == 0.0)
    {
        for (int i = 0; i < Rn; i++)
        {
            *xu += (double)mat.at<float>(i, 0);
            *yu += (double)mat.at<float>(i, 1);
        }
        *xu = (double)*xu / Rn;
        *yu = (double)*yu / Rn;
    }
    //std::cout << "  xu: " << *xu << endl;
    //std::cout << "  yu: " << *yu << endl;
    for (int i = 0; i < Rn; i++)
    {
        variance += (double)pow(mat.at<float>(i, 0) - (*xu), 2) + pow(mat.at<float>(i, 1) - (*yu), 2);
    }
    variance /= (R - K2);
    //variance = (double)variance;
    return variance;
}

double ID1func(Mat &mat, int Rn, double variance, double xu, double yu, float pj = 0.0)
{
    double ID = 0.0;
    double xsubu = 0.0;
    for (int i = 0; i < Rn; i++)
    {
        xsubu += pow(mat.at<float>(i, 0) - xu, 2) + pow(mat.at<float>(i, 1) - yu, 2);
    }
    return -log(pow(2 * Pi, 0.5) * pow(pow(variance, 0.5), 2)) - xsubu / (2 * variance);
}

double ID2func(Mat &mat1, Mat &mat2, int Rn1, int Rn2, double variance, double xu1, double yu1, double xu2, double yu2, float pj = 0.0)
{
    double ID = 0.0;
    double xsubu = 0.0;
    for (int i = 0; i < Rn1; i++)
    {
        xsubu += pow(mat1.at<float>(i, 0) - xu1, 2) + pow(mat1.at<float>(i, 1) - yu1, 2);
    }
    for (int i = 0; i < Rn2; i++)
    {
        xsubu += pow(mat2.at<float>(i, 0) - xu2, 2) + pow(mat2.at<float>(i, 1) - yu2, 2);
    }
    return -log(pow(2 * Pi, 0.5) * pow(pow(variance, 0.5), 2)) - xsubu / (2 * variance) + log(Rn1 * Rn2 / pow(Rn1 + Rn2, 2));
}

int xmeans2(Mat &mat, int *K, double xuparent = 0.0, double yuparent = 0.0)
{
    int Rn = mat.rows;
    int p = 2;
    float q =(float) p * (p + 3) / 4;
    double variance = variancefunc(mat, Rn, 1, &xuparent, &yuparent);
    double IDparent = ID1func(mat, Rn, variance, xuparent, yuparent, q);
    double BICparent = IDparent - q * log(Rn);
    printf("BICparent = %lf\n", BICparent);
    Mat labels;
    int attempts = 1;
    Mat centers(Rn, 1, mat.type());
    TermCriteria criteria{TermCriteria::COUNT, 1, 100};
    kmeans(mat, 2, labels, criteria, attempts, KMEANS_PP_CENTERS, centers);

    double xuchild1 = 0.0;
    double yuchild1 = 0.0;
    double xuchild2 = 0.0;
    double yuchild2 = 0.0;
    int Rnchild1 = 0;
    int Rnchild2 = 0;
    for (int i = 0; i < Rn; i++)
    {
        if (labels.at<int>(i) == 0)
        {
            xuchild1 += mat.at<float>(i, 0);
            yuchild1 += mat.at<float>(i, 1);
            Rnchild1++;
        }
        else
        {
            xuchild2 += mat.at<float>(i, 0);
            yuchild2 += mat.at<float>(i, 1);
            Rnchild2++;
        }
    }
    xuchild1 = xuchild1 / Rnchild1;
    yuchild1 = yuchild1 / Rnchild1;
    xuchild2 = xuchild2 / Rnchild2;
    yuchild2 = yuchild2 / Rnchild2;

    Mat matchild1(Rnchild1, 2, mat.type());
    Mat matchild2(Rnchild2, 2, mat.type());

    int temp1 = 0;
    int temp2 = 0;
    for (int i = 0; i < Rn; i++)
    {
        if (labels.at<int>(i) == 0)
        {
            matchild1.at<float>(temp1, 0) = mat.at<float>(i, 0);
            matchild1.at<float>(temp1, 1) = mat.at<float>(i, 1);
            temp1++;
        }
        if (labels.at<int>(i) == 1)
        {
            matchild2.at<float>(temp2, 0) = mat.at<float>(i, 0);
            matchild2.at<float>(temp2, 1) = mat.at<float>(i, 1);
            temp2++;
        }
    }

    double variancechild1 = variancefunc(matchild1, Rn, 2, &xuchild1, &yuchild1);
    double variancechild2 = variancefunc(matchild2, Rn, 2, &xuchild2, &yuchild2);
    variance = variancechild1 + variancechild2;
    printf("  variance = %lf, child1 = %lf, child2 = %lf\n", variance, variancechild1, variancechild2);
    q = 2 * q;
    double IDchild = ID2func(matchild1, matchild2, Rnchild1, Rnchild2, variance, xuchild1, yuchild1, xuchild2, yuchild2, q);
    double BICchild = IDchild - q * log(Rn);
    printf("  BICchild = %lf\n", BICchild);
    if (BICparent > BICchild && (*K) < 20)
    {
        (*K)++;
        printf("K changed = %d\n", *K);
        xmeans2(matchild1, K, xuchild1, yuchild1);
        xmeans2(matchild2, K, xuchild2, yuchild2);
    }
    else
    {
        printf("BIC NOT CHANGED\n");
    }
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
    Mat handimg = imread("dots5.jpg", IMREAD_COLOR);
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
    K = xmeans2(samples, &K);
    printf("result K = %d\n", K);
    //std::cout << "size: " << centers.size() << endl;
    //std::cout << "centers: " << centers << endl;
    //std::cout << "labels: " << labels.size() << endl;

    waitKey(0);
    destroyAllWindows();

    return 0;
}