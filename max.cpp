#include <vector>
#include <cmath>
#include <iostream>
#include "simplex615.hpp"
#define ZEPS 1e-10

class arbitraryFunc {
    
public: double operator() (std::vector<double>& x) {
    //f(x0,x1) = 100*(x1-x0^2)^2 + (1-x0)^2
    return 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])+(1-x[0])*(1-x[0]);
    }
};
int main(int main, char** argv) {
    double point[2] = {-1.2, 1}; // initial point to start
    arbitraryFunc foo; // WILL BE DISCUSSED LATER
    simplex615<arbitraryFunc> simplex(point, 2); // create a simplex
    simplex.amoeba(foo, 1e-7); // optimize for a function // print outputs
    std::cout << "Minimum = " << simplex.ymin() << ", at (" << simplex.xmin()[0] << ", " << simplex.xmin()[1]
    << ")" << std::endl;
    return 0;
}
