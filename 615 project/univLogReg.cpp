//
//  univLogReg.cpp
//  615 project
//
//  Created by Dylan Sun on 11/17/16.
//  Copyright © 2016 Nina Zhou. All rights reserved.
//
//  performs univariate logistic regression
//  input: data, which column the univariate predictor is in, and weights
//  output: class labels 0 or 1

#include <stdio.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "univLogReg.h"
using namespace std;


std::vector<bool> univLogReg(std::vector<double> weights, Eigen::MatrixXd &data, int column) {
    vector<bool> hypothesis;
    // data.col(data.cols()-1) is the last column, which stores the dependent variable
    // data.col(column) is the univariate predictor
    
    vector<double> beta = maximize(weights, data, column); // need to pass by ref instead
    
    double prob;
    for (int i = 0; i < data.rows(); i++) {
        prob = 1/(1 + exp(-1*(beta[0] + beta[1]*data(i, column))));
        hypothesis.push_back(prob >= 0.5);
    }
    return hypothesis;
}
