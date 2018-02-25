#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PI 3.1415926535879
#define IMG_AREA 1228800
#define MIN_SQUARE_AREA 50
#define MIN_CENTER_DIS 10
#define MIN_SQUARE_NUM 3

using namespace std;
using namespace cv;

int otsuThreshold(IplImage* img)
{
	
	int T = 0;
	int height = img->height;
	int width  = img->width;
	int step      = img->widthStep;
	int channels  = img->nChannels;
	uchar* data  = (uchar*)img->imageData;
	double gSum0;
	double gSum1;
	double N0 = 0;
	double N1 = 0;
	double u0 = 0;
	double u1 = 0;
	double w0 = 0;
	double w1 = 0;
	double u = 0;
	double tempg = -1;
	double g = -1;
	double Histogram[256]={0};// = new double[256];
	double N = width*height;
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			double temp =data[i*step + j * 3] * 0.114 + data[i*step + j * 3+1] * 0.587 + data[i*step + j * 3+2] * 0.299;
			temp = temp<0? 0:temp;
			temp = temp>255? 255:temp;
			Histogram[(int)temp]++;
		} 
	}
	
	for (int i = 0;i<256;i++)
	{
		gSum0 = 0;
		gSum1 = 0;
		N0 += Histogram[i];			
		N1 = N-N0;
		if(0==N1)break;
		w0 = N0/N;
		w1 = 1-w0;
		for (int j = 0;j<=i;j++)
		{
			gSum0 += j*Histogram[j];
		}
		u0 = gSum0/N0;
		for(int k = i+1;k<256;k++)
		{
			gSum1 += k*Histogram[k];
		}
		u1 = gSum1/N1;
		//u = w0*u0 + w1*u1;
		g = w0*w1*(u0-u1)*(u0-u1);
		if (tempg<g)
		{
			tempg = g;
			T = i;
		}
	}
	return T; 
}

void FindSquares( IplImage* src, CvSeq* squares, CvMemStorage* storage, vector<Point> &squares_centers, vector< vector<Point> > &squares_v, Point pt )
{
	CvSeq* cv_contours;    // 边缘
	CvSeq* result;         // the result of detecting squares
	CvSeqReader reader;    // the pointer to read data of "result"
	CvPoint corner[4];
	vector<Point> corner_v;
	Point temp;
	Point center;
	cvFindContours( src, storage, &cv_contours, sizeof(CvContour),CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, pt );
	while(cv_contours)
	{
		if( fabs(cvContourArea(cv_contours)) > MIN_SQUARE_AREA )   //neglect the small contours
		{
			result = cvApproxPoly( cv_contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(cv_contours)*0.02, 0 );
			if( result->total == 4  &&  cvCheckContourConvexity(result) )  
            {
				cvStartReadSeq( result, &reader, 0 );                      
                for( int i = 0; i < 4; i++ )
				{
					cvSeqPush( squares,(CvPoint*)cvGetSeqElem( result, i ));
					memcpy( corner + i, reader.ptr, result->elem_size ); 
        			CV_NEXT_SEQ_ELEM( result->elem_size, reader );
				}
				for(int i =0; i < 4; i++)    //save the corner points to corner_v, it will help us process the data
				{
					temp = corner[i];
					corner_v.push_back(temp);
				}
				center.x = (corner[0].x + corner[1].x + corner[2].x + corner[3].x) / 4;
				center.y = (corner[0].y + corner[1].y + corner[2].y + corner[3].y) / 4;
				squares_centers.push_back(center);   
				squares_v.push_back(corner_v);
				corner_v.clear();       
            } 
		}                                     
        cv_contours = cv_contours->h_next;    
	}
}

/*void DrawSquares( IplImage* img, CvSeq* squares , Point flag_center, const char* wndname)
{   
    CvSeqReader reader;   
    IplImage* cpy = cvCloneImage( img );   
    CvPoint pt[4];
    int i;       
    cvStartReadSeq( squares, &reader, 0 );     
    for( i = 0; i < squares->total; i += 4 )  
    {       
        CvPoint* rect = pt;    
        int count = 4;      
        memcpy( pt, reader.ptr, squares->elem_size ); 
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader ); 
        memcpy( pt + 1, reader.ptr, squares->elem_size );     
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );   
        memcpy( pt + 2, reader.ptr, squares->elem_size );   
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );     
        memcpy( pt + 3, reader.ptr, squares->elem_size );  
        CV_NEXT_SEQ_ELEM( squares->elem_size, reader );         
        //cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(0,255,0), 3, CV_AA, 0 );
        //cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(rand()&255,rand()&255,rand()&255), 2, CV_AA, 0 );//彩色绘制
		cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(0,255,0), 2, CV_AA, 0 );//彩色绘制
    }
	cvCircle( cpy, flag_center, 5, CV_RGB(255, 0, 0), 2);        
    cvShowImage( wndname, cpy );  
    cvReleaseImage( &cpy );
}*/

void DrawSquares( IplImage* img, vector< vector<Point> > &squares_v , Point flag_center, const char* wndname)
{   
    CvSeqReader reader;   
    IplImage* cpy = cvCloneImage( img );   
    CvPoint pt[4];           
    for( int i = 0; i < squares_v.size(); i++ )  
    {       
        CvPoint* rect = pt;    
        int count = 4;
		pt[0] = squares_v[i][0];
		pt[1] = squares_v[i][1]; 
		pt[2] = squares_v[i][2]; 
		pt[3] = squares_v[i][3];          
        //cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(0,255,0), 3, CV_AA, 0 );
        //cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(rand()&255,rand()&255,rand()&255), 2, CV_AA, 0 );//彩色绘制
		cvPolyLine( cpy, &rect, &count, 1, 1, CV_RGB(0,255,0), 2, CV_AA, 0 );//彩色绘制
    }
	cvCircle( cpy, flag_center, 5, CV_RGB(255, 0, 0), 2);        
    cvShowImage( wndname, cpy );  
    cvReleaseImage( &cpy );
}

int ComputeDistance( Point pt1, Point pt2 )
{
	int dsx = abs(pt1.x - pt2.x);
	int dsy = abs(pt1.y - pt2.y);
	return sqrt( dsx*dsx + dsy*dsy );
}

/*void CenterClassification( vector<Point> &squares_centers, vector< vector<Point> > &squares_centers_class)
{
	vector<Point> center_class;
	for(int i = 0; i < squares_centers.size(); i++)
		{
			center_class.push_back(squares_centers[i]);
			for(int j = i + 1; j < squares_centers.size(); j++)
			{
				if( ComputeDistance( squares_centers[i], squares_centers[j] ) < MIN_CENTER_DIS )
				{
					center_class.push_back(squares_centers[j]);
					squares_centers.erase(squares_centers.begin() + j - 1);
					j--;
				}
			}
			squares_centers_class.push_back(center_class);
			center_class.clear();
		}
}*/

void CenterClassification( vector<Point> &squares_centers, vector< vector<int> > &result)
{
	vector<int> centers_index;
	vector<int> result_temp;
	int index_i;
	int index_j;
	for(int i = 0; i < squares_centers.size(); i++)
	{
		centers_index.push_back(i);    //save the index of squares centers
	}

	for(int i = 0; i < centers_index.size(); i++)
		{
			result_temp.push_back(centers_index[i]);
			for(int j = i + 1; j < centers_index.size(); j++)
			{
				index_i = centers_index[i];
				index_j = centers_index[j];
				if( ComputeDistance( squares_centers[index_i], squares_centers[index_j] ) < MIN_CENTER_DIS )
				{
					result_temp.push_back(centers_index[j]);
					centers_index.erase(centers_index.begin() + j - 1);
					j--;
				}
			}
			result.push_back(result_temp);
			result_temp.clear();
		}
}

int main( int argc, char** argv )
{
    VideoCapture capture;
    capture.open(0);

    Mat gray;
    Mat binarition;
	Mat img_close;
	Mat element = getStructuringElement(MORPH_CROSS, Size(3,3));

	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Vec4i> lines;

    if (!capture.isOpened())
        return -1;

    while (1)
    {
        Mat frame;
        capture >> frame;
		//frame = imread(argv[1], 1);
        if (!frame.data)
            return -1;

		int width = frame.rows;
		int height = frame.cols;
		int img_area = width * height;

        cvtColor(frame, gray, CV_BGR2GRAY);
        IplImage frame2 = IplImage(frame);
        int adaptThresh = otsuThreshold(&frame2);
        threshold(gray, binarition, adaptThresh, 255, CV_THRESH_BINARY);
		//process_image(binarition);

		//morphologyEx(binarition, img_close, MORPH_CLOSE, element);  //close operation

		CvMemStorage* storage = cvCreateMemStorage(0);     // the storage to save contours
		CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );
		vector<Point> squares_centers;           //the center points of the squares
		vector< vector<Point> > squares_v;       //the corner points of the squares
		vector< vector<Point> > squares_useful;  //the corner points of the useful squares
		IplImage binarition2 = IplImage(binarition);
		FindSquares( &binarition2, squares, storage, squares_centers, squares_v, cvPoint(0, 0));

		vector< vector<int> > squares_class_index;
		CenterClassification(squares_centers, squares_class_index);     //classify the squares according to the center of squares

		int squares_centers_cnum = 0;
		int squares_centers_cindex;
		Point flag_center = cvPoint(0, 0);
		for(int i = 0; i < squares_class_index.size(); i++)           //find out the largest class
		{
			if( squares_class_index[i].size() > squares_centers_cnum )
			{
				squares_centers_cnum = squares_class_index[i].size();
				squares_centers_cindex = i;
			}
		}

		if( squares_centers_cnum < MIN_SQUARE_NUM )     //if the number of class members is smaller than MIN_SQUARE_NUM, there is no flag          
		{
			cout << "no flag" << endl;
			IplImage img_squares = IplImage(frame);
			cvShowImage( "Square Detection Demo", &img_squares);
			waitKey(30);
			continue;
		}

		for(int i = 0; i < squares_class_index[squares_centers_cindex].size(); i++)  //compute average of the center of squares which is our aim point
		{
			flag_center.x = flag_center.x + squares_centers[squares_class_index[squares_centers_cindex][i]].x;
			flag_center.y = flag_center.y + squares_centers[squares_class_index[squares_centers_cindex][i]].y;
		}
		flag_center.x = flag_center.x / squares_class_index[squares_centers_cindex].size();
		flag_center.y = flag_center.y / squares_class_index[squares_centers_cindex].size();

		for(int i = 0; i < squares_class_index[squares_centers_cindex].size(); i++)
		{
			squares_useful.push_back(squares_v[squares_class_index[squares_centers_cindex][i]]);
		}

		cout << flag_center << endl;

		const char* wndname = "Square Detection Demo";
		//IplImage* img_squares = cvLoadImage(argv[1]);
		IplImage img_squares = IplImage(frame);
		DrawSquares( &img_squares, squares_useful, flag_center, wndname );
		
        //imshow("show", binarition);
        waitKey(30);

    }

    return 0;
}
