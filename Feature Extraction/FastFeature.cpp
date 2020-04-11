#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cv.hpp>


using namespace cv;
using namespace std;


Mat src, src_gray ;
int thresh = 20;
int max_thresh = 255;
const char* source_window = "Source image";
const char* corners_window = "Corners detected";


void fast_demo( int, void* );

int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv, "{@input | building.jpg | input image}" );
    src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if ( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor( src, src_gray, COLOR_BGR2GRAY );

    namedWindow( source_window );
    createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, fast_demo );
    imshow( source_window, src );
    fast_demo( 0, 0 );
    waitKey();
    return 0;
}
void fast_demo( int, void* )
{

    Mat view = src_gray.clone();
    vector<KeyPoint> keypointsCorners;

    vector<Point2f> points1;

    FAST(src_gray,keypointsCorners,thresh,true); // applying FAST key point detector

    KeyPoint::convert(keypointsCorners, points1, vector<int>());

    cout<<"Point Location:"<<points1<<endl;


    // Drawing a circle around corners
    for( int i = 0; i < keypointsCorners.size(); i++ )
    {
        circle( view, keypointsCorners.at(i).pt, 5,  Scalar(0), 2, 8, 0 );
    }

    namedWindow( corners_window );
    imshow( corners_window, view );
}