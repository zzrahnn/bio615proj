//
//  adaboost.cpp
//  615 project
//
//  Created by Dylan Sun on 11/10/16.
//  Copyright © 2016 Nina Zhou. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include <cmath>
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
    vector< vector<int> > all_hypothesis;
  
    for (int t = 1; t <= T; t++) {
        // train the classifier with respect to the weighted sample set and obtain hypothesis
        std::vector<int> hypothesis = univLogReg(weights, data, t); // t is the column
        all_hypothesis.push_back(hypothesis);
        
        // calculate the training error of hypothesis
        double training_error = 0;
        bool incorrect = false;
        double delta = 0.5; // This is a given number that terminates loop if the training error is too large
        
        for (int i = 1; i < l; i++) {
            if (training_error == 0 || training_error >= delta) {
                break;
            }
            incorrect = (hypothesis[i] != data(i, T) );
            training_error += weights[i]*incorrect;
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
        for (int i = 0; i < 1; i++) {
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
        for (int t = 1; t <= T; t++) {
            sum += all_bt[t]*all_hypothesis[t][i];
        }
        f_x.push_back(sum >= 0.5);
    }

    
    return f_x;
}
