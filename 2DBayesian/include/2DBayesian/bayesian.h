#ifndef __bayesian_h__
#define __bayesian_h__

#include <map>
#include <vector>
#include <set>
#include "nav_msgs/OccupancyGrid.h"

namespace bof {

    class VelocityDistribution {
    private:
        std::map<int, float> velocityDist;

    public:
        VelocityDistribution();
        VelocityDistribution(std::map<int, float> velocityDist);
        VelocityDistribution(const int beginVelocity, const int numElems, const int stride, const int initialVelocity);

        void setVelocityProbability(const int velocity, const float probability);
        float getVelocityProbability(const int velocity);

        std::map<int, float>::const_iterator begin();
        std::map<int, float>::const_iterator end();
        void toString();

        void setVelocityDist(std::map<int, float> velocityDist);

        std::map<int, float> getVelocityDist() const;
    };

    class Cell {
    private:
        int xpos;
        int ypos;
        VelocityDistribution xVelocityDistribution;
        VelocityDistribution yVelocityDistribution;
        float occupiedProbability;
        float dt;

    private:
        void getAntecedents(std::set<Cell *>& antecedents, std::vector<std::vector<Cell> >& prevOccGrid);
        void getPrediction(float& alphaO, float& alphaE, const int xVelocity, const int yVelocity, const std::set<Cell *>& antecedents, const std::vector<std::vector<Cell> >& prevOccGrid);
        void getEstimation(std::vector<std::vector<float> >& alphaOccMatrix, std::vector<std::vector<float> >& alphaEmpMatrix, const float lvkSum);
        float getNewOccupiedProbability(const std::vector<std::vector<float> >& alphaOccMatrix);
        void updateVelocityProbabilities(const std::vector<std::vector<float> >& alphaOccMatrix, const std::vector<std::vector<float> >& alphaEmpMatrix, const std::vector<int>& xVelocityKeys, const std::vector<int>& yVelocityKeys);
        int isReachable(const int xVelocity, const int yVelocity, const Cell *cell);

    public:
        Cell(VelocityDistribution xVelocityDistribution, VelocityDistribution yVelocityDistribution, const float occupiedProbability, const int xpos, const int ypos, float dt);
        void setOccupiedProbability(const float occupiedProbability);
        int getXPos() const;
        int getYPos() const;
        float getOccupiedProbability() const;

        float getProbabilityOfXVelocity(const int xVelocity);
        float getProbabilityOfYVelocity(const int yVelocity);
        void updateDistributions(std::vector<std::vector<Cell> >& prevOccGrid, float presentOccupancy);
        void toString();
    };
}

#endif // __bayesian_h__
