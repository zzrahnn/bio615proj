
#include <Eigen/Dense>
#include <vector>
//  main.cpp
//  615 project
//
//  Created by Nina Zhou on 11/3/16.
//  Copyright © 2016 Nina Zhou. All rights reserved.
//

#include <iostream>
#include "adaboost.h"
#include "univLogReg.h"

using namespace std;

int main(int argc, const char * argv[]) {
    
    Eigen::MatrixXd testdata(10,3);
    testdata<< 10,-10,1,
                9,-9,1,
                6,5.7,0,
                8,-9.5,1,
                7.4,9.7,0,
                8,-9,1,
                10,30,0,
                7.9,-19.8,1,
                -9,-20,0,
                80,-90,1;
    std::cout<<testdata<<std::endl;
    
    
    
    std::vector<double> weights;
    double l = testdata.rows();
    double init_weight = 1/l;
    for (int i = 0; i < l; i++) {
        weights.push_back(init_weight);
    }
    
    std::vector<bool> results=univLogReg(weights, testdata, 1);
    //std::vector<bool> results = adaboost(testdata);
    std::cout << "Results: " << std::endl;
    for (int i = 0; i < results.size(); i++) {
        std::cout << results[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}

