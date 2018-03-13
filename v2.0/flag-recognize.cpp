#include "detection.h"

#define PI 3.1415926535879
#define IMG_AREA 1228800
#define MIN_SQUARE_AREA 30
#define MIN_CENTER_DIS 10
#define MIN_SQUARE_NUM 3

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    VideoCapture capture;
    capture.open(argv[1]);

	flag_detection_t *td = flag_detection_create();

	capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);

	if (!capture.isOpened())
        return -1;

    while (1)
    {
        Mat frame;
        capture >> frame;
        if (!frame.data)
            return -1;

		detect(frame, td);

		if( td->result == 0 )     //if the number of class members is smaller than MIN_SQUARE_NUM, there is no flag          
		{
			cout << "no flag" << endl;
			IplImage img_squares = IplImage(frame);
			cvShowImage( "Square Detection Demo", &img_squares);
			waitKey(30);
		}
		else
		{
			cout << td->center << endl;
			const char* wndname = "Square Detection Demo";
			IplImage img_squares = IplImage(frame);
			DrawSquares( &img_squares, td->corners, td->center, wndname);
			DrawTrajectory(td->P_w);
        	waitKey(30);
		}
		flag_detection_clear(td);
    }

	flag_detection_destroy(td);

    return 0;
}
