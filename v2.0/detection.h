#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

typedef struct flag_detection flag_detection_t;
struct flag_detection
{
    //whether there is a flag
    bool result;

    //save the corner points of the flag
    vector< vector<Point> > corners;

    //save the center point of the flag
    Point center;

    //save the distort yaml of the camera
    vector<float> distort;

    //save the 3D points of the real world
    vector<Point3f> points3d;

    //rotation matrix
    Mat R;

    //transition matrix
    Mat t;

    //real world position of camera center
    Mat P_w;
};

flag_detection_t *flag_detection_create();

void flag_detection_destroy(flag_detection_t *td);

void flag_detection_clear(flag_detection_t *td);

void detect(Mat &frame, flag_detection_t *td);

int otsuThreshold(IplImage* img);

void FindSquares( IplImage* src, CvSeq* squares, CvMemStorage* storage, vector<Point> &squares_centers, vector< vector<Point> > &squares_v, Point pt );

void DrawSquares( IplImage* img, vector< vector<Point> > &squares_v , Point flag_center, const char* wndname);

int ComputeDistance( Point pt1, Point pt2 );

void CenterClassification( vector<Point> &squares_centers, vector< vector<int> > &result);

void Read3dPoints(vector<Point3f> &points3d);

void RankPoint( vector<Point> &p, int num = 4 );

void DrawTrajectory( Mat &P );

#endif