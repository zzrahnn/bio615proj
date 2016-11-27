//
//  adaboost.cpp
//  615 project
//
//  Created by Dylan Sun on 11/10/16.
//  Copyright © 2016 Nina Zhou. All rights reserved.
//
// using univariate logistic regression as the weak classifier, run adaboost

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "adaboost.h"
#include "univLogReg.h"
using namespace std;


std::vector<bool> adaboost(Eigen::MatrixXd &data) {
    
    // initialize weights
    std::vector<double> weights;
    double l = data.rows();
    double init_weight = 1/l;
    for (int i = 0; i < l; i++) {
        weights.push_back(init_weight);
    }
    
    // Do for t = 1,..., T
    double T = data.cols()-1;
    vector<double> all_bt;
    double sum_bt = 0;
    
    // initialize object to store hypothesis; consider using a matrix or anything less stupid than this
    vector< vector<bool> > all_hypothesis;
  
    for (int t = 0; t < T; t++) {
        // train the classifier with respect to the weighted sample set and obtain hypothesis
        std::vector<bool> hypothesis = univLogReg(weights, data, t); // t is the column
        all_hypothesis.push_back(hypothesis);
        
        // calculate the training error of hypothesis
        double training_error = 0;
        bool incorrect = false;
        double delta = 0.5; // This is a given number that terminates loop if the training error is too large
        
        for (int i = 1; i < l; i++) {
            incorrect = (hypothesis[i] != data(i, T) );
            training_error += weights[i]*incorrect;
        }
        vector<bool> bad_training_error;
        if (training_error >= delta) {
            std::cout << "Training error = 0 or training error >= 0.5" << std::endl;
            return bad_training_error; // for now just return any vector of bools to get us the hell out of here
        } else if (training_error == 0) {
            return hypothesis;
        }
        // set b_t
        double b_t = log( (1-training_error)/training_error );
        all_bt.push_back(b_t);
        sum_bt += abs(b_t);
        
        // update weights
        bool correct = false;
        double sum_weights = 0;
        for (int i = 0; i < l; i++) {
            correct = (hypothesis[i] == data(i,T) );
            weights[i] = weights[i]*exp(-1*b_t*correct);
            sum_weights += weights[i];
        }
        // scale the weights; we are looping through again which might be slow? consider using "transform" function instead
        for (int i = 0; i < l; i++) {
            weights[i] = weights[i]/sum_weights;
        }
    }

    // create vector c_ts and calculate final labels; store c_ts in the all_bt vector
    std::vector<double> labels;
    
    for (int i = 0; i < all_bt.size(); i++) {
        all_bt[i] = all_bt[i]/sum_bt;
    }
    
    // calculate final probabilities and assign labels
    double sum = 0;
    vector<bool> f_x; // this stores the final labels
    for (int i = 0; i <= l; i++) {
        sum = 0;
        for (int t = 0; t < T; t++) {
            sum += all_bt[t]*all_hypothesis[t][i];
        }
        f_x.push_back(sum >= 0.5);
    }
    // Note: what exactly should we be outputting?
    // Once we hook this up to R, we should be outputting a trained classifier that can then
    // be used for predicted; e.g. we should be outputting the betas from the logistic regression
    // and all_bt (which is c_t)
    // For each new point we want to predict, we calculate the hypothesis at that point using all sets of
    // beta values (e.g. if we have 6 X columns we end up with 6 hypotheses at that point). Then, our c_t
    // tells us which hypthosis to use at that point.
    // Note that this means that our univLogReg function should probably be returning the beta's, not the
    // explicit hypotheses.

    
    return f_x;
}
