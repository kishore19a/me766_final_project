#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include <tf/transform_listener.h>

#include <stdio.h>
#include <vector>
#include <iostream>
#include <string> 
#include <sstream>
#include <algorithm>
#include <stack>
#include <math.h>

#include "/home/sine/husky_sim/devel/include/tracking/customOccupancy.h"
#include "nav_msgs/OccupancyGrid.h"
#include "std_msgs/Int8MultiArray.h"
#include "std_msgs/Float64MultiArray.h"
#include "nav_msgs/GetMap.h"
#include "tf/transform_listener.h"
#include "tf/transform_broadcaster.h"
#include "message_filters/subscriber.h"
#include "tf/message_filter.h"
#include "ros/console.h"
#include "nav_msgs/MapMetaData.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

const float res = 0.25;        //resolution of LiDAR 
const int num_values = 1080;
int counter = 0;
int clusterID = 0;
sensor_msgs::LaserScan curr;
ros::Publisher pub;
ros::Publisher pub1;



pair<float, float> convertToCart(int i, float dist)
{
	float ang = i * 0.25 * 3.14 / 180;
	float x = cos(ang) * dist;
	float y = sin(ang) * dist;

	return make_pair(x, y);
}


void addNoise(std_msgs::Float64MultiArray& map, int height, int width)
{
	int arrayForError[height+2][width+2];
	for(int i=1; i<= height; i++)
	{
		for(int j=1; j<= width; j++)
		{
			arrayForError[i][j] = map.data[(i-1)*height+(j-1)];
		}
	}
	
	for(int i=0; i<= height+1; i++)
	{
		arrayForError[0][i] = 0;
		arrayForError[height+1][i] = 0;
		arrayForError[i][0] = 0;
		arrayForError[i][height+1] = 0;
	}
   float gaussianKernel[3][3] = {{0.0256,0.1088,0.0256},{0.1088,0.4624,0.1088},{0.0256,0.1088,0.0256}};
   float arrayWithError[height][width];
   for(int i=1; i<= height; i++)
	{
		for(int j=1; j<= width; j++)
		{
			float sum = gaussianKernel[0][0]*arrayForError[i-1][j-1] + gaussianKernel[0][1]*arrayForError[i-1][j] + gaussianKernel[0][2]*arrayForError[i-1][j+1]+
			          gaussianKernel[1][0]*arrayForError[i][j-1] + gaussianKernel[1][1]*arrayForError[i][j] + gaussianKernel[1][2]*arrayForError[i][j+1]+
			          gaussianKernel[2][0]*arrayForError[i+1][j-1] + gaussianKernel[2][1]*arrayForError[i+1][j] + gaussianKernel[2][2]*arrayForError[i+1][j+1];
			arrayWithError[i][j] = sum;
			
		}
		
	}
	for(int i=1; i<= height; i++)
	{
		for(int j=1; j<= width; j++)
		{
			map.data[(i-1)*height+(j-1)] = arrayWithError[i][j];
	     }
		
	}
}


void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
	curr.header.frame_id = msg->header.frame_id;
	curr.range_min = 0.01;
	curr.range_max = 20.0;
	curr.scan_time = msg->scan_time;
	curr.angle_increment = msg->angle_increment;
	curr.angle_min = msg->angle_min;
	curr.angle_max = msg->angle_max;
	curr.time_increment = msg->time_increment;
	curr.ranges.resize(num_values);
	//curr.intensities.resize(num_values);
	curr.header.stamp = msg->header.stamp;
	curr.header.seq = msg->header.seq;
	cout<<"hey"<<endl;
	for (int i = 0; i < num_values; i++)
	{
		curr.ranges[i] = msg->ranges[i];
		//curr.intensities[i] = msg->intensities[i];
	}
    	
	vector<pair<float, float> > cart(num_values, make_pair(0, 0));
	for(int i=0; i<num_values; i++)
	{
		 cart[i] = convertToCart(i - num_values/2 , curr.ranges[i]);
		// cart[i] = convertToCart(i, curr.ranges[i]);
	}

	float reso = 0.05;
	int height = 800;
	int width = 800;
    int modified_height = 100;
	int modified_width = 100;		
	int xDist = 800;
	int yDist = 800;

	tracking::customOccupancy laserGrid;
    // nav_msgs::OccupancyGrid laserGrid;
	laserGrid.header.stamp = ros::Time::now();
	laserGrid.header.frame_id = "/lidar";
	laserGrid.info.resolution = reso;
	// The origin of the map [m, m, rad].  This is the real-world pose of the cell (0,0) in the map.
	laserGrid.info.origin.position.x = 0.0;
	laserGrid.info.origin.position.y = 0.0;
	laserGrid.info.origin.position.z = 0.0;
	laserGrid.info.origin.orientation.x = 0.0;
	laserGrid.info.origin.orientation.y = 0.0;
	laserGrid.info.origin.orientation.z = 0.0;
	laserGrid.info.origin.orientation.w = 1.0;
	laserGrid.info.width = modified_width;
	laserGrid.info.height = modified_height;
	laserGrid.info.map_load_time = ros::Time::now();

	// tracking::customOccupancy laserGrid1;
    nav_msgs::OccupancyGrid laserGrid1;
	laserGrid1.header.stamp = ros::Time::now();
	laserGrid1.header.frame_id = "/lidar";
	laserGrid1.info.resolution = reso;
	// The origin of the map [m, m, rad].  This is the real-world pose of the cell (0,0) in the map.
	laserGrid1.info.origin.position.x = 0.0;
	laserGrid1.info.origin.position.y = 0.0;
	laserGrid1.info.origin.position.z = 0.0;
	laserGrid1.info.origin.orientation.x = 0.0;
	laserGrid1.info.origin.orientation.y = 0.0;
	laserGrid1.info.origin.orientation.z = 0.0;
	laserGrid1.info.origin.orientation.w = 1.0;
	laserGrid1.info.width = width;
	laserGrid1.info.height = height;
	laserGrid1.info.map_load_time = ros::Time::now();
    
    std_msgs::Int8MultiArray map;
	// std_msgs::Float64MultiArray map;
	for(int i= width -1; i>= 0; i--)
	{
		for(int j= height - 1; j>= 0; j--)
		{
			map.data.push_back(0);
		}
	}

	for(int i=0; i<num_values; i++)
	{
		int xCoord = round(cart[i].first / reso);
		int yCoord = round(cart[i].second / reso);

        if(curr.ranges[i]!=0)
	       {
	        if(xCoord >= 0)
	        {
	        	if(yCoord >= -yDist/2 && yCoord <= yDist/2)
	        	{
	        		int x = xDist/2 - xCoord;
	        				map.data[x*width + yCoord + yDist/2] = 1;

	        	}
	        }
	    }
	}

    

	std_msgs::Float64MultiArray map_req;
	

	 for(int i=0; i<modified_width; i++)
	 {
	 	for(int j=0; j<modified_height; j++)
	 	{
	 		int a = map.data[((300+i)*width) + (350+j)];
	 		map_req.data.push_back(a);
	 	}
	 }

	 //addNoise(map, height, width);
	 addNoise(map_req, modified_height, modified_width);

	 namedWindow("occupancyGrid", CV_WINDOW_NORMAL);
	 namedWindow("occupancyGridfull", CV_WINDOW_AUTOSIZE);

     Mat mapImg = Mat::zeros(modified_height, modified_width, CV_8UC3);
     Mat mapImgfull = Mat::zeros(height, width, CV_8UC3);


 
		map_req.data[9949] = 0.5;
		     for(int i=0; i<modified_width; i++)
			 {
			 	for(int j=0; j<modified_height; j++)
			 	{
			 		 mapImg.at<Vec3b>(i, j)[0] = map_req.data[i*modified_width + j] * 255;
			 		 mapImg.at<Vec3b>(i, j)[1] = map_req.data[i*modified_width + j] * 255;
			 		 mapImg.at<Vec3b>(i, j)[2] = map_req.data[i*modified_width + j] * 255;

			 		}
			 	}

			 	for(int i=0; i<width; i++)
			 {
			 	for(int j=0; j<height; j++)
			 	{
			 		 mapImgfull.at<Vec3b>(i, j)[0] = map.data[i*width + j] * 255;
			 		 mapImgfull.at<Vec3b>(i, j)[1] = map.data[i*width + j] * 255;
			 		 mapImgfull.at<Vec3b>(i, j)[2] = map.data[i*width + j] * 255;

			 		}
			 	}
    //    while (waitKey(30) != 27) { // wait for ESC key press
    //        imshow("occupancyGrid", mapImg);
    //    }
    waitKey(1);
    imshow("occupancyGrid", mapImg);
    imshow("occupancyGridfull", mapImgfull);
    // imshow("clusteredGrid", clusterImg);
 
   
	laserGrid.data = map_req.data;
	laserGrid1.data = map.data;
	pub.publish(laserGrid);
	pub1.publish(laserGrid1);

}



int main(int argc, char **argv)
{
	ros::init(argc, argv, "OccGrd");
	ros::NodeHandle n;
	ros::NodeHandle nh;
	ros::NodeHandle nh1;
	ros::Rate loop_rate(25); 

	 // namedWindow("ClusteredGrid", CV_WINDOW_NORMAL);
	 // createTrackbar("xRadius","ClusteredGrid",&xRadius,15,lidarCallback);
    
	ros::Subscriber lidar_sub= nh.subscribe("/scan", 1, lidarCallback);
	// pub = n.advertise<nav_msgs::OccupancyGrid>("/scan/OccGrd", 1);
	pub = n.advertise<tracking::customOccupancy>("/scan/OccGrd", 1);
	pub1 = nh1.advertise<nav_msgs::OccupancyGrid>("/scan/OccGrdfull", 1);
    
	while(ros::ok())
	{
		ros::spinOnce();
		loop_rate.sleep();    
	}
	return 0;
}
