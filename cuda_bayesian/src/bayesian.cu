#include <iostream>
#include <cassert>
#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "/home/sine/husky_sim/src/cuda_bayesian/include/cuda_bayesian/defs.h"
#include "/home/sine/husky_sim/src/cuda_bayesian/include/cuda_bayesian/bayesian.cuh"
#include "/home/sine/husky_sim/src/cuda_bayesian/include/cuda_bayesian/utils.h"
#include "/home/sine/husky_sim/src/cuda_bayesian/include/cuda_bayesian/gpu_timer.h"

#include "ros/ros.h"
#include "nav_msgs/OccupancyGrid.h"
#include "sensor_msgs/JointState.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

float velocity = 0.0;

bof::Cell::Cell() {
    for (int i = 0; i < NUM_VELOCITY; ++i) {
        this->xVelocityDistribution[i] = 0;
        this->yVelocityDistribution[i] = 0;
    }
    this->yVelocityDistribution[4] = 1;
    this->occupiedProbability = 0;
    this->xpos = 0;
    this->ypos = 0;
}

__host__ __device__ bof::Cell::Cell(const float xVelocityDistribution[7], const float yVelocityDistribution[7], const float occupiedProbability, const int xpos, const int ypos) {
    for (int i = 0; i < NUM_VELOCITY; ++i) {
        this->xVelocityDistribution[i] = xVelocityDistribution[i];
        this->yVelocityDistribution[i] = yVelocityDistribution[i];
    }

    this->occupiedProbability = occupiedProbability;
    this->xpos = xpos;
    this->ypos = ypos;
}

__host__ __device__ bof::Cell** bof::Cell::getAntecedents(int *antSize, bof::Cell *prevOccGrid, float dt) {
    int stencilHalfWidth = ceil(MAX_VELOCITY * dt);
    *antSize = ((-stencilHalfWidth + ypos > 0 ? stencilHalfWidth : ypos) + (stencilHalfWidth + ypos < GRID_ROWS ? stencilHalfWidth : GRID_ROWS - ypos - 1) + 1) *
        ((-stencilHalfWidth + xpos > 0 ? stencilHalfWidth : xpos) + (stencilHalfWidth + xpos < GRID_COLS ? stencilHalfWidth : GRID_COLS - xpos - 1) + 1);

    Cell **antecedents = new Cell*[*antSize];

    int counter = 0;
    for (int i = -stencilHalfWidth; i <= stencilHalfWidth; ++i) {
        for (int j = -stencilHalfWidth; j <= stencilHalfWidth; ++j) {
            if (i + ypos >= 0 && i + ypos < GRID_ROWS && j + xpos >= 0 && j + xpos < GRID_COLS) {
                antecedents[counter++] = &prevOccGrid[(i + ypos) * GRID_ROWS + (j + xpos)];
            }
        }
    }

    // printf("stencilHalfWidth: %d, dt: %f, Pos: (%d, %d): antSize: %d\n", stencilHalfWidth, dt, xpos, ypos, *antSize);
    return antecedents;
}

__host__ __device__ void bof::Cell::getPrediction(float *alphaO, float *alphaE, const int xVelocity, const int yVelocity, bof::Cell **antecedents, const int antSize, const bof::Cell *prevOccGrid, float dt) {
    *alphaO = *alphaE = 0.0f;

    for (int i = 0; i < antSize; ++i) {
        Cell *cell = antecedents[i];

        *alphaO += (1.0f / antSize) *
            cell->xVelocityDistribution[(xVelocity + MAX_VELOCITY) / VEL_STRIDE] * cell->yVelocityDistribution[(yVelocity + MAX_VELOCITY) / VEL_STRIDE] *
            cell->isReachable(xVelocity, yVelocity, cell, dt) *
            cell->occupiedProbability;

        *alphaE += (1.0f / antSize) *
            cell->xVelocityDistribution[(xVelocity + MAX_VELOCITY) / VEL_STRIDE] * cell->yVelocityDistribution[(yVelocity + MAX_VELOCITY) / VEL_STRIDE] *
            cell->isReachable(xVelocity, yVelocity, cell, dt) *
            (1.0f - cell->occupiedProbability);
    }
}

__host__ __device__ void bof::Cell::getEstimation(float *alphaOccMatrix, float *alphaEmpMatrix, const float lvkSum) {
    for (int i = 0; i < NUM_VELOCITY; ++i) {
        for (int j = 0; j < NUM_VELOCITY; ++j) {
            alphaOccMatrix[i * NUM_VELOCITY + j] /= lvkSum;
            alphaEmpMatrix[i * NUM_VELOCITY + j] /= lvkSum;
        }
    }
}

__host__ __device__ float bof::Cell::getNewOccupiedProbability(const float *alphaOccMatrix) {
    float sum = 0.0f;
    for (int i = 0; i < NUM_VELOCITY; ++i) {
        for (int j = 0; j < NUM_VELOCITY; ++j) {
            sum += alphaOccMatrix[i * NUM_VELOCITY + j];
        }
    }

    return sum;
}

__host__ __device__ void bof::Cell::updateVelocityProbabilities(const float *alphaOccMatrix, const float *alphaEmpMatrix) {
    for (int i = 0; i < NUM_VELOCITY; ++i) {
        for (int j = 0; j < NUM_VELOCITY; ++j) {
            xVelocityDistribution[j] += alphaOccMatrix[i * NUM_VELOCITY + j] + alphaEmpMatrix[i * NUM_VELOCITY + j];
            yVelocityDistribution[i] += alphaOccMatrix[i * NUM_VELOCITY + j] + alphaEmpMatrix[i * NUM_VELOCITY + j];
        }
    }
}

__host__ __device__ int bof::Cell::isReachable(const int xVelocity, const int yVelocity, const bof::Cell *cell, float dt) {
    int reachableXPos = lroundf(cell->xpos + xVelocity * dt);
    int reachableYPos = lroundf(cell->ypos + yVelocity * dt);

    if (xpos == reachableXPos && ypos == reachableYPos)
        return 1;

    return 0;
}

__host__ __device__ void bof::Cell::updateDistributions(bof::Cell *prevOccGrid, float dt) {
    assert(dt > 0);

    int antSize;
    Cell **antecedents = getAntecedents(&antSize, prevOccGrid, dt);
    assert(antSize > 0);

    float *betaOccMatrix = new float[NUM_VELOCITY * NUM_VELOCITY];
    float *betaEmpMatrix = new float[NUM_VELOCITY * NUM_VELOCITY];
    float lvkSum = 0;

    for (int xVel = -MAX_VELOCITY; xVel <= MAX_VELOCITY; xVel += VEL_STRIDE) {
        int i = (xVel + MAX_VELOCITY) / VEL_STRIDE;
        for (int yVel = -MAX_VELOCITY; yVel <= MAX_VELOCITY; yVel += VEL_STRIDE) {
            int j = (yVel + MAX_VELOCITY) / VEL_STRIDE;
            float alphaO = 0, alphaE = 0;
            getPrediction(&alphaO, &alphaE, xVel, yVel, antecedents, antSize, prevOccGrid, dt);

            float betaO = xVelocityDistribution[i] * yVelocityDistribution[j] * alphaO;
            float betaE = xVelocityDistribution[i] * yVelocityDistribution[j] * alphaE;
            lvkSum += betaO + betaE;

            betaOccMatrix[i * NUM_VELOCITY + j] = betaO;
            betaEmpMatrix[i * NUM_VELOCITY + j] = betaE;
        }
    }

    if (!lvkSum) {
        getEstimation(betaOccMatrix, betaEmpMatrix, lvkSum);
        occupiedProbability = getNewOccupiedProbability(betaOccMatrix);
        updateVelocityProbabilities(betaOccMatrix, betaEmpMatrix);
    }

    delete betaOccMatrix;
    delete betaEmpMatrix;
    delete antecedents;
}

void bof::Cell::toString() {
    std::cout << "Pos: (" << xpos << ", " << ypos << "), ";

    std::cout << "xVel: [" << xVelocityDistribution[0];
    for (int i = 1; i < NUM_VELOCITY; ++i)
        std::cout << ", " << xVelocityDistribution[i];

    std::cout << "], yVel: [" << yVelocityDistribution[0];
    for (int i = 1; i < NUM_VELOCITY; ++i)
        std::cout << ", " << yVelocityDistribution[i];

    std::cout << "], Occ: " << occupiedProbability << std::endl;
}

__global__ void computeDistributions(bof::Cell *occGrid, bof::Cell *prevOccGrid, float dt) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_COLS || y >= GRID_ROWS) {
        return;
    }

    unsigned int index = y * GRID_ROWS + x;
    occGrid[index].updateDistributions(prevOccGrid, dt);
}

void callKernel(bof::Cell *occGrid, bof::Cell *prevOccGrid, float dt) {
    /* Compute gridsize and blocksize */
    const unsigned int width = 32;
    const dim3 blockSize(width, width, 1);

    unsigned int gridCols = GRID_COLS / width + (GRID_COLS % width != 0);
    unsigned int gridRows = GRID_ROWS / width + (GRID_ROWS % width != 0);
    const dim3 gridSize(gridRows, gridCols, 1);

    computeDistributions<<<gridSize, blockSize>>>(occGrid, prevOccGrid, dt);
}

void printOccupancy(bof::Cell *m) {
    for (int i = 0; i < GRID_ROWS; ++i) {
        for (int j = 0; j < GRID_COLS; ++j) {
            m[i * GRID_ROWS + j].toString();
        }
    }
}


void chatterCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg){
    size_t numBytes = sizeof(bof::Cell) * GRID_ROWS * GRID_COLS;

    bof::Cell *h_m = new bof::Cell[GRID_ROWS * GRID_COLS];
    bof::Cell *d_prevOccGrid;
    bof::Cell *d_occGrid;

    cout << "Size of grid (bytes): " << numBytes << endl;

    for (int i = 0; i < GRID_ROWS; ++i) {
        for (int j = 0; j < GRID_COLS; ++j) {
            h_m[i * GRID_ROWS + j].xpos = j;
            h_m[i * GRID_ROWS + j].ypos = i;
            h_m[i * GRID_ROWS + j].xVelocityDistribution[NUM_VELOCITY / 2] = 1;
            h_m[i * GRID_ROWS + j].yVelocityDistribution[NUM_VELOCITY / 2] = 1;
            h_m[i * GRID_ROWS + j].occupiedProbability = msg->data[i*GRID_ROWS+j];
        }
    }
    // printOccupancy(h_m);
    
    Mat map_without_bayesian = Mat::zeros(GRID_ROWS, GRID_COLS, CV_8UC3);
     for (int i = 0; i < GRID_ROWS; ++i)
	{
	    for (int j = 0; j < GRID_COLS; ++j)
		{
			int a = h_m[i * GRID_ROWS + j].occupiedProbability;
			    if(a == 0 || a == 1)
			    {
				map_without_bayesian.at<Vec3b>(i, j)[0] = a * 255;
			 	map_without_bayesian.at<Vec3b>(i, j)[1] = a * 255;
			 	map_without_bayesian.at<Vec3b>(i, j)[2] = a * 255;
			 }
			}
	}

    checkCudaErrors((cudaMalloc(&d_prevOccGrid, numBytes)));
    checkCudaErrors(cudaMemcpy(d_prevOccGrid, h_m, numBytes, cudaMemcpyHostToDevice));
    checkCudaErrors((cudaMalloc(&d_occGrid, numBytes)));
    checkCudaErrors(cudaMemcpy(d_occGrid, h_m, numBytes, cudaMemcpyHostToDevice));

    /* Call kernel */
    GpuTimer timer;
    timer.Start();
    float dt = 0.5f; // compute time delay here
    callKernel(d_occGrid, d_prevOccGrid, dt);
    timer.Stop();
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    cout << "Elapsed time: " << timer.Elapsed() << endl;
    checkCudaErrors(cudaMemcpy(h_m, d_occGrid, numBytes, cudaMemcpyDeviceToHost));

    Mat map_with_bayesian = Mat::zeros(GRID_ROWS, GRID_COLS, CV_8UC3);
     for (int i = 0; i < GRID_ROWS; ++i)
	{
	    for (int j = 0; j < GRID_COLS; ++j)
		{
			int a = h_m[i * GRID_ROWS + j].occupiedProbability;
			    if(a == 0 || a == 1)
			    {
				map_with_bayesian.at<Vec3b>(i, j)[0] = a * 255;
			 	map_with_bayesian.at<Vec3b>(i, j)[1] = a * 255;
			 	map_with_bayesian.at<Vec3b>(i, j)[2] = a * 255;
			 }
			}
	}

    waitKey(1);
    imshow("map_with_bayesian", map_with_bayesian);
    imshow("map_without_bayesian", map_without_bayesian);

    // printOccupancy(h_m);
    /* cleanup */
    cudaFree(d_prevOccGrid);
    cudaFree(d_occGrid);
    delete h_m;

}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "cuda_map");

  ros::NodeHandle n;
  ros::NodeHandle nh;
  ros::NodeHandle nh1;
  ros::Rate loop_rate(25); 

  namedWindow("map_without_bayesian", CV_WINDOW_NORMAL);
  namedWindow("map_with_bayesian", CV_WINDOW_NORMAL);
  
  ros::Subscriber sub = n.subscribe("/scan/fusedOccGrd", 1, chatterCallback);

  while(ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();    
    }
  return 0;
}

