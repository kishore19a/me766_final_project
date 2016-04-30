#include <iostream>
#include <vector>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "nav_msgs/OccupancyGrid.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/Float64.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "math.h"
#include "/home/sine/husky_sim/devel/include/tracking/customOccupancy.h"


#include "bayesian.h"

#define GRIDROWS 400
#define GRIDCOLS 800

using namespace std;
using namespace cv;

float angle = 0.0;
float velocity = 0.0;
float currtime = 0.0;
float prevtime = 0.0;
float dt = 0.0;

void displayOccupancyGrid(vector<vector<bof::Cell> >& occGrid) {

    Mat mapImg = Mat::zeros(GRIDROWS, GRIDCOLS, CV_8UC3);
    
    for (int i = 0; i < GRIDROWS; ++i) {
        for (int j = 0; j < GRIDCOLS; ++j) {
          float a = occGrid[i][j].getOccupiedProbability();
            mapImg.at<Vec3b>(i, j)[0] = a * 255;
            mapImg.at<Vec3b>(i, j)[1] = a * 255;
            mapImg.at<Vec3b>(i, j)[2] = a * 255;
        }
    }

    //    while (waitKey(30) != 27) { // wait for ESC key press
    //        imshow("occupancyGrid", mapImg);
    //    }
    waitKey(1);
    imshow("occupancyGrid", mapImg);
}

void getDirection(const std_msgs::Float64::ConstPtr& msg){
   angle = msg->data;
}

void getVelocity(const sensor_msgs::JointState::ConstPtr& msg){
   velocity = msg->velocity[0];
   	currtime = msg->header.stamp.sec + 0.000000001*msg->header.stamp.nsec;

   if(currtime - prevtime < 10)
   	dt = currtime - prevtime;

   prevtime = currtime;
   //cout<<"dt= "<<dt<<endl;
}

void chatterCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg){
    cout<<"hi"<<endl;
    int start_s=clock();
    
    int occupancy=0;
    float velocityX = velocity*cos(angle);
    float velocityY = velocity*sin(angle);
    bof::VelocityDistribution xVelDist(-3, 7, 1, 0);
    bof::VelocityDistribution yVelDist(-3, 7, 1, 0);

    // yVelDist.setVelocityProbability(0, 0.1);
    yVelDist.setVelocityProbability(round((velocityY*100)/5), 1.0);
    // xVelDist.setVelocityProbability(round(velocityX)*10, 1.0);

    // yVelDist.setVelocityProbability(-4, 0.1);

    /* Initialize occupancy grid */
    vector<vector<bof::Cell> > occGrid;
    for (int y = 0; y < GRIDROWS; ++y) {
        vector<bof::Cell> occRow;
        for (int x = 0; x < GRIDCOLS; ++x) {
        	occupancy = msg->data[y*GRIDROWS+x];
            bof::Cell cell(xVelDist, yVelDist, occupancy, x, y, dt);
            occRow.push_back(cell);
        }
        occGrid.push_back(occRow);
    }

    // for (int k = 0; k < 50; ++k) {
        /* Update occupancy grid */
        vector<vector<bof::Cell> > prevOccGrid = occGrid;

        for (int i = 0; i < GRIDROWS; ++i) {
            for (int j = 0; j < GRIDCOLS; ++j) {
                occGrid[i][j].updateDistributions(prevOccGrid,msg->data[i*GRIDROWS+j]);
                // cout << "[" << i << "][" << j << "]: ";
                // occGrid[i][j].toString();
            }
        }

        displayOccupancyGrid(occGrid);
    // }

  int stop_s=clock();
  cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)<< endl;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "listener");

  ros::NodeHandle n;
  ros::NodeHandle nh;
  ros::NodeHandle nh1;
  namedWindow("occupancyGrid", CV_WINDOW_NORMAL);
  
  ros::Subscriber sub2 = nh1.subscribe("/imu_data1", 1 , getDirection);
  ros::Subscriber sub1 = nh.subscribe("/joint_states", 1 , getVelocity);
  ros::Subscriber sub = n.subscribe("/scan/fusedOccGrd", 1, chatterCallback);

  ros::spin();

  return 0;
}