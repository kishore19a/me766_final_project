#include <iostream>
#include <cassert>
#include <cmath>

#include "bayesian.h"
#include "utils.h"
#include "nav_msgs/OccupancyGrid.h"

#define GRIDROWS 400
#define GRIDCOLS 800

namespace bof {

    /* VelocityDistribution - function definitions */
    VelocityDistribution::VelocityDistribution() {
    }

    VelocityDistribution::VelocityDistribution(std::map<int, float> velocityDist) {
        this->velocityDist = velocityDist;
    }

    VelocityDistribution::VelocityDistribution(const int beginVelocity, const int numElems, const int stride, const int initialVelocity) {
        velocityDist.clear();
        for (int i = 0, velocity = beginVelocity; i < numElems; ++i) {
            velocityDist[velocity] = 0;
            velocity += stride;
        }

        assert(velocityDist.size() == numElems);
        assert(velocityDist.find(initialVelocity) != velocityDist.end());

        velocityDist[initialVelocity] = 1;
    }

    void VelocityDistribution::setVelocityProbability(const int velocity, const float probability) {
        assert(velocityDist.find(velocity) != velocityDist.end());
        velocityDist[velocity] = probability;
    }

    float VelocityDistribution::getVelocityProbability(const int velocity) {
        assert(velocityDist.find(velocity) != velocityDist.end());
        return velocityDist[velocity];
    }

    void VelocityDistribution::setVelocityDist(std::map<int, float> velocityDist) {
        this->velocityDist = velocityDist;
    }

    std::map<int, float> VelocityDistribution::getVelocityDist() const {
        return velocityDist;
    }

    std::map<int, float>::const_iterator VelocityDistribution::begin() {
        return velocityDist.begin();
    }

    std::map<int, float>::const_iterator VelocityDistribution::end() {
        return velocityDist.end();
    }

    void VelocityDistribution::toString() {
        printMap(velocityDist);
        std::cout << std::endl;
    }

    /* Cell - function definitions */
    void Cell::getAntecedents(std::set<Cell *>& antecedents, std::vector<std::vector<Cell> >& prevOccGrid) {
        assert(prevOccGrid.size() != 0 && prevOccGrid[0].size() != 0);

        for (std::map<int, float>::const_iterator xit = xVelocityDistribution.begin(); xit != xVelocityDistribution.end(); ++xit) {
            for (std::map<int, float>::const_iterator yit = yVelocityDistribution.begin(); yit != yVelocityDistribution.end(); ++yit) {
                int x = round(xpos + xit->first * dt);
                int y = round(ypos + yit->first * dt);

                if (y >= 0 && y < prevOccGrid.size()) {
                    if (x >= 0 && x < prevOccGrid[y].size()) {
                        antecedents.insert(&prevOccGrid[y][x]); // TODO: check
                    }
                }
            }
        }
    }

    void Cell::getPrediction(float& alphaO, float& alphaE, const int xVelocity, const int yVelocity, const std::set<Cell *>& antecedents, const std::vector<std::vector<Cell> >& prevOccGrid) {
        alphaO = 0;
        alphaE = 0;

        for (std::set<Cell *>::const_iterator it = antecedents.begin(); it != antecedents.end(); ++it) {
            Cell *cell = *it;
            alphaO += (1.0 / antecedents.size()) * // P(A)
                    cell->getProbabilityOfXVelocity(xVelocity) * cell->getProbabilityOfYVelocity(yVelocity) * // P(V|A)
                    isReachable(xVelocity, yVelocity, cell) * // P(C|V,A)
                    cell->getOccupiedProbability(); // P(occ^-1|A)

            alphaE += (1.0 / antecedents.size()) * // P(A)
                    cell->getProbabilityOfXVelocity(xVelocity) * cell->getProbabilityOfYVelocity(yVelocity) * // P(V|A)
                    isReachable(xVelocity, yVelocity, cell) * // P(C|V,A)
                    (1.0 - cell->getOccupiedProbability()); // P(emp^-1|A)
        }
    }

    void Cell::getEstimation(std::vector<std::vector<float> >& alphaOccMatrix, std::vector<std::vector<float> >& alphaEmpMatrix, const float lvkSum) {
        assert(alphaOccMatrix.size() != 0 && alphaEmpMatrix.size() != 0 && alphaOccMatrix.size() == alphaEmpMatrix.size());
        assert(alphaOccMatrix[0].size() != 0 && alphaEmpMatrix[0].size() != 0);
        for (int i = 0; i < alphaOccMatrix.size(); ++i) {
            for (int j = 0; j < alphaOccMatrix[i].size(); ++j) {
                alphaOccMatrix[i][j] /= lvkSum;
                alphaEmpMatrix[i][j] /= lvkSum;
            }
        }
    }

    float Cell::getNewOccupiedProbability(const std::vector<std::vector<float> >& alphaOccMatrix) {
        assert(alphaOccMatrix.size() != 0 && alphaOccMatrix[0].size() != 0);

        float sum = 0;
        for (int i = 0; i < alphaOccMatrix.size(); ++i) {
            for (int j = 0; j < alphaOccMatrix[i].size(); ++j) {
                sum += alphaOccMatrix[i][j];
            }
        }

        return sum;
    }

    void Cell::updateVelocityProbabilities(const std::vector<std::vector<float> >& alphaOccMatrix, const std::vector<std::vector<float> >& alphaEmpMatrix, const std::vector<int>& xVelocityKeys, const std::vector<int>& yVelocityKeys) {
        assert(alphaOccMatrix.size() != 0 && alphaEmpMatrix.size() != 0 && alphaOccMatrix.size() == alphaEmpMatrix.size());
        assert(alphaOccMatrix[0].size() != 0 && alphaEmpMatrix[0].size() != 0);
        assert(xVelocityKeys.size() == alphaOccMatrix.size());
        assert(yVelocityKeys.size() != 0);

        for (int i = 0; i < alphaOccMatrix.size(); ++i) {
            float newXVelocityProbability = 0;
            for (int j = 0; j < alphaOccMatrix[i].size(); ++j) {
                newXVelocityProbability += alphaOccMatrix[i][j] + alphaEmpMatrix[i][j];
            }
            xVelocityDistribution.setVelocityProbability(xVelocityKeys[i], newXVelocityProbability);
        }

        for (int i = 0; i < alphaOccMatrix[0].size(); ++i) {
            float newYVelocityProbability = 0;
            for (int j = 0; j < alphaOccMatrix.size(); ++j) {
                newYVelocityProbability += alphaOccMatrix[j][i] + alphaEmpMatrix[j][i];
            }
            yVelocityDistribution.setVelocityProbability(yVelocityKeys[i], newYVelocityProbability);
        }
    }

    int Cell::isReachable(const int xVelocity, const int yVelocity, const Cell* cell) {
        int reachableXPos = round(cell->getXPos() + xVelocity * dt);
        int reachableYPos = round(cell->getYPos() + yVelocity * dt);

        if (xpos == reachableXPos && ypos == reachableYPos)
            return 1;

        return 0;
    }

    Cell::Cell(VelocityDistribution xVelocityDistribution, VelocityDistribution yVelocityDistribution, float occupiedProbability, const int xpos, const int ypos, float dt) {
        this->xpos = xpos;
        this->ypos = ypos;
        this->xVelocityDistribution = xVelocityDistribution;
        this->yVelocityDistribution = yVelocityDistribution;
        this->occupiedProbability = occupiedProbability;
        this->dt = dt;
    }

    void Cell::setOccupiedProbability(const float occupiedProbability) {
        this->occupiedProbability = occupiedProbability;
    }

    int Cell::getXPos() const {
        return xpos;
    }

    int Cell::getYPos() const {
        return ypos;
    }

    float Cell::getOccupiedProbability() const {
        return occupiedProbability;
    }

    float Cell::getProbabilityOfXVelocity(const int xVelocity) {
        return xVelocityDistribution.getVelocityProbability(xVelocity);
    }

    float Cell::getProbabilityOfYVelocity(const int yVelocity) {
        return yVelocityDistribution.getVelocityProbability(yVelocity);
    }

    void Cell::updateDistributions(std::vector<std::vector<Cell> >& prevOccGrid, float presentOccupancy) {
        std::set<Cell *> antecedents;
        getAntecedents(antecedents, prevOccGrid);

        assert(antecedents.size() > 0);

        std::vector<int> xVelocityKeys;
        
        std::vector<int> yVelocityKeys;
        bool yVelocityKeysUpdated = false;

        std::vector<std::vector<float> > betaOccMatrix;
        std::vector<std::vector<float> > betaEmpMatrix;

        float lvkSum = 0;
        for (std::map<int, float>::const_iterator xit = xVelocityDistribution.begin(); xit != xVelocityDistribution.end(); ++xit) {
            xVelocityKeys.push_back(xit->first);
            std::vector<float> betaOccRow;
            std::vector<float> betaEmpRow;

            for (std::map<int, float>::const_iterator yit = yVelocityDistribution.begin(); yit != yVelocityDistribution.end(); ++yit) {
                if (!yVelocityKeysUpdated)
                    yVelocityKeys.push_back(yit->first);

                float alphaO = 0;
                float alphaE = 0;
                getPrediction(alphaO, alphaE, xit->first, yit->first, antecedents, prevOccGrid);

                float betaO = xit->second * yit->second * alphaO * presentOccupancy;
                float betaE = xit->second * yit->second * alphaE * presentOccupancy;
                lvkSum += betaO + betaE;

                betaOccRow.push_back(betaO);
                betaEmpRow.push_back(betaE);
            }
            betaOccMatrix.push_back(betaOccRow);
            betaEmpMatrix.push_back(betaEmpRow);
            yVelocityKeysUpdated = true;
        }

        // assert(lvkSum != 0);

        if (lvkSum != 0) {
            getEstimation(betaOccMatrix, betaEmpMatrix, lvkSum);
            occupiedProbability = getNewOccupiedProbability(betaOccMatrix);
            updateVelocityProbabilities(betaOccMatrix, betaEmpMatrix, xVelocityKeys, yVelocityKeys);
        } else {
            occupiedProbability = presentOccupancy;
            bof::VelocityDistribution xVelDist(-6, 7, 2, 0);
            bof::VelocityDistribution yVelDist(-6, 7, 2, 0);

            yVelDist.setVelocityProbability(0, 0.1);
            yVelDist.setVelocityProbability(-2, 0.8);
            yVelDist.setVelocityProbability(-4, 0.1);

            xVelocityDistribution = xVelDist;
            yVelocityDistribution = yVelDist;
        }
    }

    void Cell::toString() {
        std::cout << "Occupied Probability: " << occupiedProbability << std::endl;
        //        xVelocityDistribution.toString();
        //        yVelocityDistribution.toString();
    }
}