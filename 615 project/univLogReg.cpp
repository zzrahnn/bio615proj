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
#include "simplex615.hpp"
using namespace std;

#define ZEPS 1e-10

class likelihoodFunc {
    public:
    double operator() (std::vector<double>& beta, Eigen::MatrixXd &data, int column, std::vector<double> weights) {
        int y = data.cols()-1;
        double likelihood =  0;
        // data(i,y) is the same as saying "Y_i"
        // data(i, column) is the same as saying "X_i"; note the index starts at 0 and not 1 for column
        for (int i = 0; i < data.rows(); i++) {
            likelihood -= weights[i]*(data(i, y)*(beta[0]+beta[1]*data(i, column)-log(1+exp(beta[0]+beta[1]*data(i, column)))) - (1-data(i,y))*(log(1+exp(beta[0]+beta[1]*data(i, column)))));
        }
    return likelihood;
    }
};

std::vector<bool> univLogReg(std::vector<double> weights, Eigen::MatrixXd &data, int column) {
    vector<bool> hypothesis;
    // data.col(data.cols()-1) is the last column, which stores the dependent variable
    // data.col(column) is the univariate predictor
    
    double point[2] = {0, 0}; // initial point to start
    likelihoodFunc foo; // WILL BE DISCUSSED LATER
    simplex615<likelihoodFunc> simplex(point, 2, data, column, weights); // create a simplex
    simplex.amoeba(foo, 1e-7); // optimize for a function
    
    vector<double> beta;
    beta.push_back(simplex.xmin()[0]);
    beta.push_back(simplex.xmin()[1]);
    
    double prob;
    for (int i = 0; i < data.rows(); i++) {
        prob = 1/(1 + exp(-1*(beta[0] + beta[1]*data(i, column))));
        hypothesis.push_back(prob >= 0.5);
    }
    return hypothesis;
}
