#ifndef __bayesian_cuh__
#define __bayesian_cuh__

#include <cuda_runtime.h>

#include "defs.h"

namespace bof {

    class Cell {
    public:
        int xpos;
        int ypos;
        float xVelocityDistribution[NUM_VELOCITY];
        float yVelocityDistribution[NUM_VELOCITY];
        float occupiedProbability;

        /**
         * Default Constructor. Sets everything to 0;
         * To be called by host only.
         */
        Cell();

        /**
         * Constructor
         *
         * @param xVelocityDistribution     velocity distribution along x-axis
         * @param yVelocityDistribution     velocity distribution along y-axis
         * @param occupiedProbability       probability of occupancy
         * @param xpos                      x-position of cell
         * @param ypos                      y-position of cell
         */
        __host__ __device__ Cell(const float xVelocityDistribution[NUM_VELOCITY], const float yVelocityDistribution[NUM_VELOCITY], const float occupiedProbability, const int xpos, const int ypos);

        /**
         * Returns the set of antecedents for the current cell
         *
         * @param antSize       size of 'antecedents' array
         * @param prevOccGrid   linearized 2D array of previous occupancy grid
         * @param dt            time difference between previous scan and new scan
         *
         * @return              an array of 'Cell *' (Cell pointers which are antecedents)
         */
        __host__ __device__ bof::Cell** getAntecedents(int *antSize, Cell *prevOccGrid, float dt);
        
        /**
         * Computes 'alphaO' and 'alphaE' for a given velocity
         *
         * @param alphaO        pointer to computed alphaO value
         * @param alphaE        pointer to computed alphaE value
         * @param xVelocity     velocity in x-axis
         * @param yVelocity     velocity in y-axis
         * @param antecedents   array of antecedents for the current cell (array of Cell*)
         * @param antSize       size of 'antecedents'
         * @param prevOccGrid   linearized 2D array of previous occupancy grid
         * @param dt            time difference between previous scan and new scan
         */
        __host__ __device__ void getPrediction(float *alphaO, float *alphaE, const int xVelocity, const int yVelocity, Cell **antecedents, const int antSize, const Cell *prevOccGrid, float dt);

        /**
         * Normalizes 'alphaOccMatrix' and 'alphaEmpMatrix'
         *
         * @param alphaOccMatrix    linearized 2D array of alphaOccMatrix
         * @param alphaEmpMatrix    linearized 2D array of alphaEmpMatrix
         * @param lvkSum            normalization divisor
         */
        __host__ __device__ void getEstimation(float *alphaOccMatrix, float *alphaEmpMatrix, const float lvkSum);

        /**
         * Computes the new occupancy probability of the cell
         *
         * @param alphaOccMatrix    linearized 2D array of alphaOccMatrix
         */
        __host__ __device__ float getNewOccupiedProbability(const float *alphaOccMatrix);

        /**
         * Computes the new velocity distribution of the cell
         *
         * @param alphaOccMatrix    linearized 2D array of alphaOccMatrix
         * @param alphaEmpMatrix    linearized 2D array of alphaEmpMatrix
         */
        __host__ __device__ void updateVelocityProbabilities(const float *alphaOccMatrix, const float *alphaEmpMatrix);

        /**
         * Computes if the current cell is reachable from a given 'cell' with given velocity
         *
         * @param xVelocity     velocity in x-axis
         * @param xVelocity     velocity in x-axis
         * @param cell          previous cell
         * @param dt            time difference between previous scan and new scan
         */
        __host__ __device__ int isReachable(const int xVelocity, const int yVelocity, const Cell *cell, float dt);

        /**
         * Parent function for updating occupancy grid from previous occupancy grid
         *
         * @param prevOccGrid   linearized 2D array of previous occupancy grid
         * @param dt            time difference between previous scan and new scan
         */
        __host__ __device__ void updateDistributions(Cell *prevOccGrid, float dt);

        /**
         * Generic print function for debug purposes
         */
        void toString();
    };

}

/**
 * Kernel for computing distribution of each cell
 *
 * @param occGrid       linearized 2D array of measured occupancy grid
 * @param prevOccGrid   linearized 2D array of previous occupancy grid
 * @param dt            time difference between previous scan and new scan
 */
__global__ void computeDistributions(bof::Cell *occGrid, bof::Cell *prevOccGrid, float dt);

/**
* Wrapper for kernel call
*
* @param occGrid       linearized 2D array of measured occupancy grid
* @param prevOccGrid   linearized 2D array of previous occupancy grid
* @param dt            time difference between previous scan and new scan
*/
void callKernel(bof::Cell *occGrid, bof::Cell *prevOccGrid, float dt);

#endif // __bayesian_cuh__
