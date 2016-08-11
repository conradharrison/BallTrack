#pragma comment(lib, "opencv_core310d.lib")
#pragma comment(lib, "opencv_highgui310d.lib")
#pragma comment(lib, "opencv_imgproc310d.lib")
#pragma comment(lib, "opencv_imgcodecs310d.lib")
#pragma comment(lib, "opencv_objdetect310d.lib")
#pragma comment(lib, "opencv_videoio310d.lib")

#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

//callback for trackbar. nothing to be done here     
void on_trackbar( int, void* )
{
}

int main(int argc, char* argv[])
{
        int t1min=0,t1max=0,t2min=0,t2max=0,t3min=0,t3max=0; // other variables used

		VideoCapture capture(0);

		if (!capture.isOpened())
		{
			std::cout << "Fatal: Cannot open the video camera" << std::endl;
			return -1;
		}

        // grab an image from the capture
		Mat frame;
		bool bSuccess = capture.read(frame);

		if (!bSuccess)
		{
			std::cout << "Warning: Cannot read a frame from video stream" << std::endl;
		}

        // Create a window in which the captured images will be presented
        namedWindow( "Camera", CV_WINDOW_AUTOSIZE );
        namedWindow( "HSV", CV_WINDOW_AUTOSIZE );
        namedWindow( "F1", CV_WINDOW_AUTOSIZE );
        namedWindow( "F2", CV_WINDOW_AUTOSIZE );
        namedWindow( "F3", CV_WINDOW_AUTOSIZE );
        //cvNamedWindow( "EdgeDetection", CV_WINDOW_AUTOSIZE );

        /// Create Trackbars
        char TrackbarName1[50]="t1min";
        char TrackbarName2[50]="t1max";
        char TrackbarName3[50]="t2min";
        char TrackbarName4[50]="t2max";
        char TrackbarName5[50]="t3min";
        char TrackbarName6[50]="t3max";

        createTrackbar( TrackbarName1, "F1", &t1min, 260 , NULL );
        createTrackbar( TrackbarName2, "F1", &t1max, 260,  NULL  );

        createTrackbar( TrackbarName3, "F2", &t2min, 260 , NULL );
        createTrackbar( TrackbarName4, "F2", &t2max, 260,  NULL  );

        createTrackbar( TrackbarName5, "F3", &t3min, 260 , NULL );
        createTrackbar( TrackbarName6, "F3", &t3max, 260,  NULL  );

        // Load threshold from the slider bars in these 2 parameters
        Scalar hsv_min(t1min, t2min, t3min, 0);
        Scalar hsv_max(t1max, t2max ,t3max, 0);

        // get the image data
		Size capture_size = frame.size();
        //int step = frame.step;

        // Initialize different images that are going to be used in the program
        Mat hsv_frame(capture_size, CV_8UC3); // image converted to HSV plane
		std::vector<Mat> thresholded_channels;
		Mat thresholded(capture_size, CV_8UC1); // final thresholded image
		Mat thresholded1(capture_size, CV_8UC1); // Component image threshold
		Mat thresholded2(capture_size, CV_8UC1);
		Mat thresholded3(capture_size, CV_8UC1);
		Mat filtered(capture_size, CV_8UC1);  //smoothed image

        while( 1 )
        {   
                // Load threshold from the slider bars in these 2 parameters
                hsv_min = Scalar(t1min, t2min, t3min, 0);
                hsv_max = Scalar(t1max, t2max ,t3max, 0);

                // Get one frame
				bool bSuccess = capture.read(frame);

				if (!bSuccess)
				{
					std::cout << "Warning: Cannot read a frame from video stream" << std::endl;
				}

                // Covert color space to HSV as it is much easier to filter colors in the HSV color-space.
                cvtColor(frame, hsv_frame, CV_BGR2HSV);

                // Filter out colors which are out of range.
                inRange(hsv_frame, hsv_min, hsv_max, thresholded);

                // the below lines of code is for visual purpose only remove after calibration 
                //--------------FROM HERE-----------------------------------
                //Split image into its 3 one dimensional images
				split(hsv_frame, thresholded_channels);

                // Filter out colors which are out of range.
				inRange(thresholded_channels[0], Scalar(t1min, 0, 0, 0), Scalar(t1max, 0, 0, 0), thresholded_channels[0]);
				inRange(thresholded_channels[1], Scalar(t2min, 0, 0, 0), Scalar(t2max, 0, 0, 0), thresholded_channels[1]);
				inRange(thresholded_channels[2], Scalar(t3min, 0, 0, 0), Scalar(t3max, 0, 0, 0), thresholded_channels[2]);

                //-------------REMOVE OR COMMENT AFTER CALIBRATION TILL HERE ------------------

                // hough detector works better with some smoothing of the image
				GaussianBlur(thresholded, thresholded, Size(5, 5), 5, 5);

                //hough transform to detect circle
				std::vector<Vec3f> circles;
                HoughCircles(thresholded, circles, CV_HOUGH_GRADIENT, 2, thresholded.size().height/4, 100, 50, 50, 400);

                for (int i = 0; i < circles.size(); i++)
                {   
					//get the parameters of circles detected
					Vec3f p = circles.at(i);
                    std::cout << "Ball! x=" << p[0] <<" y=" << p[1] <<" r=" << p[2] << "\n\r";
                    // draw a circle with the centre and the radius obtained from the hough transform
                    circle( frame, Point(cvRound(p[0]),cvRound(p[1])), 2, CV_RGB(255,255,255),-1, 8, 0 );
                    circle( frame, Point(cvRound(p[0]),cvRound(p[1])), cvRound(p[2]), CV_RGB(0,255,0), 2, 8, 0 );
                }

                /* for testing purpose you can show all the images but when done with calibration 
                   only show frame to keep the screen clean  */  

                imshow( "Camera", frame ); // Original stream with detected ball overlay
                imshow( "HSV", hsv_frame); // Original stream in the HSV color space
                imshow( "After Color Filtering", thresholded ); // The stream after color filtering
				imshow("F1", thresholded_channels[0]); // individual filters
				imshow("F2", thresholded_channels[1]);
				imshow("F3", thresholded_channels[2]);
                //imshow( "filtered", thresholded );

                //If ESC key pressed, Key=0x10001B under OpenCV 0.9.7(linux version),
                //remove higher bits using AND operator
                if( (cvWaitKey(10) & 255) == 27 ) break;
        }
        
        cvDestroyWindow( "mywindow" );
        return 0;
}

