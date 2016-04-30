#include <iostream>

#include "utils.h"

void printMap(const std::map<int, float>& intFloatMap) {
    std::cout << "{";
    for (std::map<int, float>::const_iterator it = intFloatMap.begin(); it != intFloatMap.end(); ++it) {
        if (it != intFloatMap.begin())
            std::cout << ", ";
        std::cout << it->first << ": " << it->second;
    }
    std::cout << "}";
}
