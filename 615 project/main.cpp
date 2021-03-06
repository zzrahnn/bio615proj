
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
    
    Eigen::MatrixXd testdata(756,3);
    testdata<<
    29.00     ,     0     ,     1     ,
    2.00     ,     0     ,     0     ,
    30.00     ,     1     ,     0     ,
    25.00     ,     0     ,     0     ,
    0.92     ,     1     ,     1     ,
    47.00     ,     1     ,     1     ,
    63.00     ,     0     ,     1     ,
    39.00     ,     1     ,     0     ,
    58.00     ,     0     ,     1     ,
    71.00     ,     1     ,     0     ,
    47.00     ,     1     ,     0     ,
    19.00     ,     0     ,     1     ,
    50.00     ,     0     ,     1     ,
    24.00     ,     1     ,     0     ,
    36.00     ,     1     ,     0     ,
    37.00     ,     1     ,     1     ,
    47.00     ,     0     ,     1     ,
    26.00     ,     1     ,     1     ,
    25.00     ,     1     ,     0     ,
    25.00     ,     1     ,     1     ,
    19.00     ,     0     ,     1     ,
    28.00     ,     1     ,     1     ,
    45.00     ,     1     ,     0     ,
    39.00     ,     1     ,     1     ,
    30.00     ,     0     ,     1     ,
    58.00     ,     0     ,     1     ,
    45.00     ,     0     ,     1     ,
    22.00     ,     0     ,     1     ,
    41.00     ,     1     ,     0     ,
    48.00     ,     1     ,     0     ,
    44.00     ,     0     ,     1     ,
    59.00     ,     0     ,     1     ,
    60.00     ,     0     ,     1     ,
    45.00     ,     1     ,     0     ,
    53.00     ,     0     ,     1     ,
    58.00     ,     0     ,     1     ,
    36.00     ,     1     ,     1     ,
    33.00     ,     1     ,     0     ,
    36.00     ,     1     ,     1     ,
    36.00     ,     0     ,     1     ,
    14.00     ,     0     ,     1     ,
    11.00     ,     1     ,     1     ,
    49.00     ,     1     ,     0     ,
    36.00     ,     1     ,     0     ,
    46.00     ,     1     ,     0     ,
    47.00     ,     0     ,     1     ,
    27.00     ,     1     ,     1     ,
    31.00     ,     0     ,     1     ,
    27.00     ,     1     ,     0     ,
    26.00     ,     0     ,     1     ,
    64.00     ,     0     ,     1     ,
    37.00     ,     1     ,     0     ,
    39.00     ,     0     ,     1     ,
    55.00     ,     0     ,     1     ,
    70.00     ,     1     ,     0     ,
    69.00     ,     0     ,     1     ,
    36.00     ,     0     ,     1     ,
    39.00     ,     1     ,     0     ,
    38.00     ,     0     ,     1     ,
    27.00     ,     1     ,     1     ,
    31.00     ,     1     ,     0     ,
    27.00     ,     0     ,     1     ,
    31.00     ,     1     ,     1     ,
    17.00     ,     0     ,     1     ,
    4.00     ,     1     ,     1     ,
    27.00     ,     0     ,     1     ,
    50.00     ,     1     ,     0     ,
    48.00     ,     0     ,     1     ,
    49.00     ,     1     ,     1     ,
    48.00     ,     0     ,     1     ,
    39.00     ,     1     ,     0     ,
    23.00     ,     0     ,     1     ,
    53.00     ,     0     ,     1     ,
    36.00     ,     0     ,     0     ,
    30.00     ,     1     ,     0     ,
    24.00     ,     0     ,     1     ,
    19.00     ,     1     ,     0     ,
    28.00     ,     0     ,     1     ,
    23.00     ,     0     ,     1     ,
    64.00     ,     1     ,     0     ,
    60.00     ,     0     ,     1     ,
    49.00     ,     1     ,     1     ,
    44.00     ,     1     ,     1     ,
    22.00     ,     0     ,     1     ,
    60.00     ,     1     ,     1     ,
    48.00     ,     0     ,     1     ,
    37.00     ,     1     ,     0     ,
    35.00     ,     0     ,     1     ,
    47.00     ,     1     ,     0     ,
    22.00     ,     0     ,     1     ,
    45.00     ,     0     ,     1     ,
    49.00     ,     1     ,     1     ,
    71.00     ,     1     ,     0     ,
    54.00     ,     1     ,     1     ,
    38.00     ,     1     ,     0     ,
    19.00     ,     0     ,     1     ,
    58.00     ,     0     ,     1     ,
    45.00     ,     0     ,     1     ,
    23.00     ,     1     ,     1     ,
    46.00     ,     1     ,     0     ,
    25.00     ,     1     ,     1     ,
    21.00     ,     0     ,     1     ,
    48.00     ,     1     ,     1     ,
    49.00     ,     0     ,     1     ,
    45.00     ,     1     ,     0     ,
    36.00     ,     0     ,     1     ,
    55.00     ,     1     ,     0     ,
    52.00     ,     0     ,     1     ,
    24.00     ,     0     ,     1     ,
    16.00     ,     0     ,     1     ,
    44.00     ,     0     ,     1     ,
    51.00     ,     0     ,     1     ,
    42.00     ,     1     ,     0     ,
    35.00     ,     0     ,     1     ,
    35.00     ,     1     ,     1     ,
    38.00     ,     1     ,     1     ,
    35.00     ,     0     ,     1     ,
    50.00     ,     0     ,     0     ,
    49.00     ,     1     ,     1     ,
    46.00     ,     1     ,     0     ,
    58.00     ,     1     ,     0     ,
    41.00     ,     1     ,     0     ,
    42.00     ,     1     ,     1     ,
    40.00     ,     0     ,     1     ,
    42.00     ,     1     ,     0     ,
    55.00     ,     0     ,     1     ,
    50.00     ,     0     ,     1     ,
    16.00     ,     0     ,     1     ,
    29.00     ,     1     ,     0     ,
    21.00     ,     0     ,     1     ,
    30.00     ,     1     ,     0     ,
    15.00     ,     0     ,     1     ,
    30.00     ,     1     ,     0     ,
    46.00     ,     1     ,     0     ,
    54.00     ,     1     ,     0     ,
    36.00     ,     1     ,     1     ,
    28.00     ,     1     ,     0     ,
    65.00     ,     1     ,     0     ,
    33.00     ,     0     ,     1     ,
    44.00     ,     1     ,     0     ,
    37.00     ,     0     ,     1     ,
    55.00     ,     1     ,     0     ,
    47.00     ,     1     ,     0     ,
    36.00     ,     1     ,     0     ,
    58.00     ,     1     ,     0     ,
    31.00     ,     0     ,     1     ,
    23.00     ,     0     ,     1     ,
    19.00     ,     0     ,     1     ,
    64.00     ,     1     ,     0     ,
    64.00     ,     1     ,     0     ,
    22.00     ,     0     ,     1     ,
    28.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    18.00     ,     1     ,     0     ,
    17.00     ,     0     ,     1     ,
    52.00     ,     1     ,     1     ,
    46.00     ,     1     ,     0     ,
    56.00     ,     0     ,     1     ,
    43.00     ,     0     ,     1     ,
    31.00     ,     1     ,     0     ,
    33.00     ,     0     ,     1     ,
    27.00     ,     0     ,     1     ,
    55.00     ,     1     ,     0     ,
    54.00     ,     0     ,     1     ,
    61.00     ,     1     ,     0     ,
    48.00     ,     0     ,     1     ,
    18.00     ,     0     ,     1     ,
    13.00     ,     1     ,     1     ,
    21.00     ,     0     ,     1     ,
    34.00     ,     1     ,     1     ,
    40.00     ,     0     ,     1     ,
    36.00     ,     1     ,     1     ,
    50.00     ,     1     ,     0     ,
    39.00     ,     0     ,     1     ,
    56.00     ,     1     ,     1     ,
    28.00     ,     1     ,     1     ,
    56.00     ,     1     ,     0     ,
    56.00     ,     1     ,     0     ,
    24.00     ,     1     ,     0     ,
    18.00     ,     0     ,     1     ,
    24.00     ,     1     ,     1     ,
    23.00     ,     0     ,     1     ,
    45.00     ,     1     ,     1     ,
    40.00     ,     0     ,     1     ,
    6.00     ,     1     ,     1     ,
    57.00     ,     1     ,     0     ,
    32.00     ,     1     ,     1     ,
    62.00     ,     1     ,     0     ,
    54.00     ,     1     ,     1     ,
    43.00     ,     0     ,     1     ,
    52.00     ,     0     ,     1     ,
    62.00     ,     0     ,     1     ,
    67.00     ,     1     ,     0     ,
    63.00     ,     0     ,     0     ,
    61.00     ,     1     ,     0     ,
    46.00     ,     0     ,     1     ,
    52.00     ,     1     ,     0     ,
    39.00     ,     0     ,     1     ,
    18.00     ,     0     ,     1     ,
    48.00     ,     1     ,     1     ,
    49.00     ,     1     ,     0     ,
    39.00     ,     0     ,     1     ,
    17.00     ,     1     ,     1     ,
    46.00     ,     1     ,     0     ,
    31.00     ,     1     ,     1     ,
    61.00     ,     1     ,     0     ,
    47.00     ,     1     ,     0     ,
    64.00     ,     1     ,     0     ,
    60.00     ,     0     ,     1     ,
    60.00     ,     1     ,     0     ,
    55.00     ,     0     ,     1     ,
    54.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    57.00     ,     1     ,     0     ,
    45.00     ,     0     ,     1     ,
    31.00     ,     0     ,     1     ,
    50.00     ,     1     ,     0     ,
    50.00     ,     0     ,     1     ,
    27.00     ,     1     ,     0     ,
    20.00     ,     0     ,     1     ,
    51.00     ,     1     ,     0     ,
    21.00     ,     1     ,     1     ,
    36.00     ,     0     ,     1     ,
    40.00     ,     1     ,     0     ,
    32.00     ,     1     ,     0     ,
    33.00     ,     1     ,     0     ,
    30.00     ,     1     ,     0     ,
    28.00     ,     0     ,     1     ,
    18.00     ,     1     ,     0     ,
    34.00     ,     1     ,     0     ,
    32.00     ,     0     ,     1     ,
    57.00     ,     1     ,     0     ,
    18.00     ,     1     ,     0     ,
    23.00     ,     1     ,     0     ,
    36.00     ,     0     ,     1     ,
    28.00     ,     1     ,     0     ,
    51.00     ,     1     ,     0     ,
    32.00     ,     1     ,     1     ,
    19.00     ,     0     ,     1     ,
    28.00     ,     1     ,     0     ,
    36.00     ,     0     ,     1     ,
    4.00     ,     0     ,     1     ,
    1.00     ,     1     ,     1     ,
    12.00     ,     0     ,     1     ,
    34.00     ,     1     ,     1     ,
    19.00     ,     0     ,     1     ,
    23.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    27.00     ,     1     ,     0     ,
    15.00     ,     0     ,     1     ,
    45.00     ,     1     ,     0     ,
    40.00     ,     0     ,     1     ,
    20.00     ,     0     ,     1     ,
    25.00     ,     1     ,     0     ,
    36.00     ,     0     ,     1     ,
    25.00     ,     1     ,     0     ,
    42.00     ,     0     ,     1     ,
    26.00     ,     1     ,     1     ,
    26.00     ,     0     ,     1     ,
    0.83     ,     1     ,     1     ,
    31.00     ,     0     ,     1     ,
    19.00     ,     1     ,     0     ,
    54.00     ,     1     ,     0     ,
    44.00     ,     0     ,     0     ,
    52.00     ,     1     ,     0     ,
    30.00     ,     1     ,     0     ,
    30.00     ,     0     ,     0     ,
    29.00     ,     1     ,     0     ,
    29.00     ,     1     ,     0     ,
    27.00     ,     1     ,     0     ,
    24.00     ,     1     ,     1     ,
    35.00     ,     1     ,     0     ,
    31.00     ,     0     ,     1     ,
    8.00     ,     0     ,     1     ,
    22.00     ,     0     ,     0     ,
    30.00     ,     0     ,     0     ,
    20.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    49.00     ,     0     ,     1     ,
    8.00     ,     1     ,     1     ,
    28.00     ,     0     ,     1     ,
    18.00     ,     1     ,     0     ,
    28.00     ,     1     ,     0     ,
    22.00     ,     0     ,     1     ,
    25.00     ,     1     ,     0     ,
    18.00     ,     1     ,     0     ,
    32.00     ,     0     ,     1     ,
    18.00     ,     0     ,     1     ,
    42.00     ,     1     ,     0     ,
    34.00     ,     0     ,     1     ,
    8.00     ,     1     ,     1     ,
    23.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    19.00     ,     1     ,     0     ,
    38.00     ,     1     ,     0     ,
    38.00     ,     0     ,     0     ,
    35.00     ,     1     ,     0     ,
    35.00     ,     1     ,     0     ,
    38.00     ,     1     ,     0     ,
    24.00     ,     0     ,     1     ,
    16.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    45.00     ,     1     ,     0     ,
    24.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    34.00     ,     1     ,     0     ,
    30.00     ,     1     ,     0     ,
    50.00     ,     1     ,     0     ,
    30.00     ,     1     ,     0     ,
    23.00     ,     0     ,     1     ,
    1.00     ,     1     ,     1     ,
    44.00     ,     1     ,     0     ,
    28.00     ,     1     ,     0     ,
    6.00     ,     0     ,     1     ,
    30.00     ,     1     ,     1     ,
    43.00     ,     1     ,     0     ,
    45.00     ,     0     ,     1     ,
    7.00     ,     0     ,     1     ,
    24.00     ,     0     ,     1     ,
    24.00     ,     0     ,     1     ,
    49.00     ,     1     ,     0     ,
    48.00     ,     0     ,     1     ,
    34.00     ,     1     ,     0     ,
    32.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    18.00     ,     0     ,     0     ,
    53.00     ,     0     ,     1     ,
    23.00     ,     1     ,     0     ,
    21.00     ,     0     ,     1     ,
    52.00     ,     1     ,     0     ,
    42.00     ,     1     ,     0     ,
    36.00     ,     0     ,     1     ,
    21.00     ,     1     ,     0     ,
    41.00     ,     1     ,     1     ,
    33.00     ,     1     ,     0     ,
    17.00     ,     0     ,     1     ,
    23.00     ,     0     ,     1     ,
    34.00     ,     1     ,     0     ,
    22.00     ,     0     ,     0     ,
    45.00     ,     0     ,     1     ,
    31.00     ,     1     ,     0     ,
    30.00     ,     1     ,     0     ,
    26.00     ,     0     ,     0     ,
    34.00     ,     0     ,     1     ,
    26.00     ,     1     ,     0     ,
    22.00     ,     0     ,     1     ,
    1.00     ,     0     ,     1     ,
    3.00     ,     0     ,     1     ,
    25.00     ,     1     ,     0     ,
    48.00     ,     1     ,     0     ,
    57.00     ,     0     ,     0     ,
    2.00     ,     1     ,     1     ,
    27.00     ,     1     ,     0     ,
    19.00     ,     0     ,     1     ,
    30.00     ,     1     ,     0     ,
    20.00     ,     1     ,     0     ,
    45.00     ,     1     ,     0     ,
    46.00     ,     1     ,     0     ,
    41.00     ,     0     ,     1     ,
    13.00     ,     0     ,     1     ,
    19.00     ,     1     ,     1     ,
    30.00     ,     1     ,     0     ,
    48.00     ,     1     ,     0     ,
    71.00     ,     1     ,     0     ,
    54.00     ,     1     ,     0     ,
    64.00     ,     1     ,     0     ,
    32.00     ,     1     ,     0     ,
    18.00     ,     0     ,     1     ,
    2.00     ,     1     ,     1     ,
    32.00     ,     1     ,     0     ,
    3.00     ,     1     ,     1     ,
    26.00     ,     1     ,     0     ,
    19.00     ,     1     ,     0     ,
    20.00     ,     1     ,     1     ,
    29.00     ,     0     ,     1     ,
    39.00     ,     1     ,     0     ,
    22.00     ,     1     ,     1     ,
    24.00     ,     1     ,     0     ,
    28.00     ,     1     ,     0     ,
    50.00     ,     0     ,     1     ,
    20.00     ,     1     ,     0     ,
    40.00     ,     1     ,     0     ,
    42.00     ,     0     ,     1     ,
    21.00     ,     1     ,     0     ,
    32.00     ,     0     ,     1     ,
    34.00     ,     1     ,     0     ,
    33.00     ,     0     ,     1     ,
    2.00     ,     0     ,     1     ,
    8.00     ,     0     ,     1     ,
    36.00     ,     1     ,     0     ,
    34.00     ,     1     ,     0     ,
    30.00     ,     0     ,     1     ,
    28.00     ,     0     ,     1     ,
    23.00     ,     1     ,     0     ,
    0.80     ,     1     ,     1     ,
    25.00     ,     0     ,     1     ,
    3.00     ,     1     ,     1     ,
    50.00     ,     0     ,     1     ,
    21.00     ,     0     ,     1     ,
    25.00     ,     0     ,     1     ,
    18.00     ,     0     ,     1     ,
    20.00     ,     0     ,     1     ,
    30.00     ,     0     ,     1     ,
    59.00     ,     1     ,     0     ,
    30.00     ,     0     ,     1     ,
    35.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    25.00     ,     1     ,     0     ,
    41.00     ,     1     ,     0     ,
    25.00     ,     1     ,     0     ,
    14.00     ,     1     ,     0     ,
    50.00     ,     0     ,     1     ,
    22.00     ,     1     ,     0     ,
    27.00     ,     0     ,     1     ,
    29.00     ,     1     ,     0     ,
    27.00     ,     0     ,     0     ,
    30.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    35.00     ,     0     ,     1     ,
    30.00     ,     1     ,     0     ,
    28.00     ,     0     ,     1     ,
    23.00     ,     1     ,     0     ,
    12.00     ,     0     ,     1     ,
    40.00     ,     0     ,     1     ,
    36.00     ,     0     ,     1     ,
    28.00     ,     1     ,     0     ,
    32.00     ,     0     ,     1     ,
    29.00     ,     0     ,     1     ,
    4.00     ,     0     ,     1     ,
    2.00     ,     1     ,     1     ,
    36.00     ,     1     ,     0     ,
    33.00     ,     0     ,     1     ,
    32.00     ,     1     ,     1     ,
    26.00     ,     0     ,     1     ,
    30.00     ,     1     ,     0     ,
    24.00     ,     0     ,     1     ,
    18.00     ,     1     ,     0     ,
    42.00     ,     1     ,     0     ,
    13.00     ,     1     ,     0     ,
    16.00     ,     1     ,     0     ,
    35.00     ,     0     ,     1     ,
    16.00     ,     0     ,     1     ,
    25.00     ,     1     ,     1     ,
    18.00     ,     0     ,     1     ,
    20.00     ,     1     ,     1     ,
    30.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    40.00     ,     0     ,     0     ,
    24.00     ,     1     ,     0     ,
    41.00     ,     1     ,     0     ,
    18.00     ,     0     ,     1     ,
    0.83     ,     1     ,     1     ,
    23.00     ,     1     ,     0     ,
    20.00     ,     1     ,     0     ,
    25.00     ,     1     ,     0     ,
    35.00     ,     1     ,     0     ,
    17.00     ,     1     ,     0     ,
    32.00     ,     1     ,     0     ,
    20.00     ,     1     ,     0     ,
    39.00     ,     1     ,     0     ,
    39.00     ,     0     ,     0     ,
    6.00     ,     0     ,     0     ,
    2.00     ,     0     ,     0     ,
    17.00     ,     0     ,     1     ,
    38.00     ,     0     ,     0     ,
    9.00     ,     0     ,     0     ,
    26.00     ,     1     ,     0     ,
    11.00     ,     0     ,     0     ,
    4.00     ,     1     ,     0     ,
    20.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    25.00     ,     1     ,     0     ,
    18.00     ,     0     ,     0     ,
    24.00     ,     1     ,     0     ,
    35.00     ,     1     ,     0     ,
    40.00     ,     1     ,     0     ,
    38.00     ,     0     ,     1     ,
    5.00     ,     1     ,     0     ,
    9.00     ,     1     ,     0     ,
    3.00     ,     1     ,     1     ,
    13.00     ,     1     ,     0     ,
    23.00     ,     1     ,     1     ,
    5.00     ,     0     ,     1     ,
    45.00     ,     0     ,     1     ,
    23.00     ,     1     ,     0     ,
    17.00     ,     0     ,     0     ,
    27.00     ,     1     ,     0     ,
    23.00     ,     1     ,     0     ,
    20.00     ,     1     ,     0     ,
    32.00     ,     1     ,     0     ,
    33.00     ,     0     ,     1     ,
    3.00     ,     0     ,     1     ,
    18.00     ,     0     ,     1     ,
    40.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    15.00     ,     0     ,     1     ,
    45.00     ,     0     ,     0     ,
    18.00     ,     0     ,     0     ,
    27.00     ,     0     ,     0     ,
    22.00     ,     1     ,     0     ,
    19.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    20.00     ,     1     ,     0     ,
    32.00     ,     1     ,     1     ,
    21.00     ,     1     ,     0     ,
    18.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    6.00     ,     1     ,     0     ,
    9.00     ,     0     ,     0     ,
    40.00     ,     1     ,     0     ,
    32.00     ,     0     ,     0     ,
    26.00     ,     1     ,     0     ,
    18.00     ,     0     ,     1     ,
    20.00     ,     0     ,     0     ,
    29.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    35.00     ,     1     ,     0     ,
    21.00     ,     1     ,     1     ,
    20.00     ,     0     ,     0     ,
    19.00     ,     1     ,     0     ,
    18.00     ,     0     ,     0     ,
    18.00     ,     1     ,     0     ,
    38.00     ,     1     ,     0     ,
    30.00     ,     1     ,     0     ,
    17.00     ,     1     ,     0     ,
    21.00     ,     0     ,     0     ,
    21.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    24.00     ,     1     ,     0     ,
    33.00     ,     1     ,     0     ,
    33.00     ,     1     ,     0     ,
    28.00     ,     1     ,     0     ,
    16.00     ,     0     ,     1     ,
    37.00     ,     0     ,     0     ,
    28.00     ,     1     ,     0     ,
    24.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    32.00     ,     1     ,     0     ,
    29.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    18.00     ,     1     ,     0     ,
    20.00     ,     1     ,     0     ,
    19.00     ,     1     ,     1     ,
    24.00     ,     1     ,     0     ,
    24.00     ,     1     ,     0     ,
    36.00     ,     1     ,     0     ,
    31.00     ,     1     ,     0     ,
    31.00     ,     1     ,     0     ,
    30.00     ,     0     ,     0     ,
    22.00     ,     0     ,     1     ,
    43.00     ,     1     ,     0     ,
    35.00     ,     1     ,     0     ,
    27.00     ,     1     ,     0     ,
    19.00     ,     1     ,     0     ,
    30.00     ,     1     ,     0     ,
    36.00     ,     0     ,     1     ,
    3.00     ,     1     ,     1     ,
    9.00     ,     1     ,     1     ,
    59.00     ,     1     ,     0     ,
    19.00     ,     1     ,     0     ,
    44.00     ,     1     ,     0     ,
    17.00     ,     0     ,     1     ,
    45.00     ,     1     ,     1     ,
    22.00     ,     0     ,     0     ,
    19.00     ,     1     ,     0     ,
    29.00     ,     1     ,     1     ,
    30.00     ,     0     ,     1     ,
    34.00     ,     1     ,     0     ,
    28.00     ,     0     ,     0     ,
    0.33     ,     1     ,     0     ,
    27.00     ,     1     ,     0     ,
    25.00     ,     1     ,     0     ,
    24.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    17.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    33.00     ,     0     ,     1     ,
    1.00     ,     1     ,     1     ,
    0.17     ,     0     ,     1     ,
    25.00     ,     1     ,     0     ,
    36.00     ,     1     ,     1     ,
    36.00     ,     0     ,     1     ,
    30.00     ,     1     ,     1     ,
    23.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    19.00     ,     0     ,     1     ,
    65.00     ,     1     ,     0     ,
    42.00     ,     1     ,     0     ,
    43.00     ,     1     ,     0     ,
    32.00     ,     1     ,     0     ,
    19.00     ,     1     ,     1     ,
    30.00     ,     0     ,     1     ,
    24.00     ,     0     ,     0     ,
    23.00     ,     0     ,     1     ,
    24.00     ,     0     ,     1     ,
    24.00     ,     1     ,     1     ,
    23.00     ,     1     ,     0     ,
    22.00     ,     0     ,     1     ,
    18.00     ,     1     ,     0     ,
    16.00     ,     1     ,     0     ,
    45.00     ,     1     ,     0     ,
    47.00     ,     1     ,     0     ,
    5.00     ,     0     ,     1     ,
    21.00     ,     0     ,     0     ,
    18.00     ,     1     ,     0     ,
    9.00     ,     0     ,     0     ,
    48.00     ,     0     ,     0     ,
    16.00     ,     1     ,     0     ,
    25.00     ,     1     ,     0     ,
    38.00     ,     0     ,     1     ,
    22.00     ,     1     ,     0     ,
    16.00     ,     0     ,     1     ,
    33.00     ,     1     ,     0     ,
    9.00     ,     1     ,     1     ,
    41.00     ,     1     ,     0     ,
    38.00     ,     1     ,     0     ,
    40.00     ,     1     ,     0     ,
    43.00     ,     0     ,     0     ,
    14.00     ,     1     ,     0     ,
    16.00     ,     0     ,     0     ,
    9.00     ,     1     ,     0     ,
    10.00     ,     0     ,     0     ,
    6.00     ,     1     ,     0     ,
    11.00     ,     1     ,     0     ,
    40.00     ,     1     ,     0     ,
    32.00     ,     1     ,     0     ,
    20.00     ,     1     ,     0     ,
    37.00     ,     1     ,     0     ,
    28.00     ,     1     ,     0     ,
    19.00     ,     1     ,     0     ,
    24.00     ,     0     ,     0     ,
    17.00     ,     0     ,     0     ,
    28.00     ,     1     ,     0     ,
    24.00     ,     0     ,     1     ,
    20.00     ,     1     ,     0     ,
    41.00     ,     1     ,     0     ,
    45.00     ,     0     ,     1     ,
    26.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    27.00     ,     1     ,     1     ,
    18.00     ,     0     ,     0     ,
    26.00     ,     0     ,     1     ,
    23.00     ,     0     ,     0     ,
    22.00     ,     0     ,     1     ,
    28.00     ,     0     ,     0     ,
    22.00     ,     0     ,     1     ,
    2.00     ,     0     ,     0     ,
    43.00     ,     1     ,     0     ,
    27.00     ,     0     ,     1     ,
    42.00     ,     1     ,     0     ,
    27.00     ,     0     ,     0     ,
    25.00     ,     0     ,     0     ,
    27.00     ,     1     ,     1     ,
    19.00     ,     0     ,     1     ,
    20.00     ,     1     ,     0     ,
    48.00     ,     1     ,     0     ,
    17.00     ,     1     ,     0     ,
    34.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    33.00     ,     1     ,     0     ,
    32.00     ,     1     ,     1     ,
    29.00     ,     1     ,     0     ,
    26.00     ,     1     ,     1     ,
    49.00     ,     1     ,     0     ,
    1.00     ,     0     ,     1     ,
    33.00     ,     1     ,     0     ,
    4.00     ,     1     ,     1     ,
    24.00     ,     0     ,     0     ,
    19.00     ,     1     ,     0     ,
    32.00     ,     1     ,     1     ,
    27.00     ,     1     ,     0     ,
    21.00     ,     0     ,     0     ,
    32.00     ,     1     ,     1     ,
    20.00     ,     0     ,     0     ,
    17.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    30.00     ,     1     ,     0     ,
    21.00     ,     1     ,     1     ,
    23.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    4.00     ,     0     ,     1     ,
    39.00     ,     1     ,     0     ,
    20.00     ,     1     ,     0     ,
    21.00     ,     0     ,     1     ,
    44.00     ,     1     ,     0     ,
    42.00     ,     1     ,     0     ,
    21.00     ,     0     ,     1     ,
    24.00     ,     1     ,     0     ,
    25.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    39.00     ,     1     ,     1     ,
    26.00     ,     0     ,     0     ,
    4.00     ,     0     ,     1     ,
    22.00     ,     0     ,     0     ,
    26.00     ,     1     ,     0     ,
    1.50     ,     0     ,     0     ,
    36.00     ,     0     ,     0     ,
    18.00     ,     1     ,     0     ,
    25.00     ,     1     ,     1     ,
    37.00     ,     0     ,     0     ,
    22.00     ,     0     ,     1     ,
    20.00     ,     1     ,     0     ,
    26.00     ,     1     ,     1     ,
    29.00     ,     1     ,     0     ,
    29.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    32.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    21.00     ,     0     ,     0     ,
    36.00     ,     1     ,     0     ,
    39.00     ,     1     ,     0     ,
    25.00     ,     0     ,     0     ,
    45.00     ,     0     ,     0     ,
    36.00     ,     1     ,     0     ,
    30.00     ,     0     ,     0     ,
    20.00     ,     1     ,     1     ,
    23.00     ,     0     ,     0     ,
    21.00     ,     1     ,     0     ,
    1.50     ,     0     ,     0     ,
    25.00     ,     1     ,     1     ,
    18.00     ,     0     ,     1     ,
    63.00     ,     0     ,     1     ,
    18.00     ,     0     ,     0     ,
    31.00     ,     1     ,     0     ,
    31.00     ,     0     ,     0     ,
    15.00     ,     1     ,     0     ,
    28.00     ,     1     ,     0     ,
    36.00     ,     1     ,     0     ,
    28.00     ,     1     ,     0     ,
    10.00     ,     0     ,     0     ,
    36.00     ,     1     ,     0     ,
    30.00     ,     0     ,     0     ,
    22.00     ,     1     ,     1     ,
    29.00     ,     1     ,     0     ,
    47.00     ,     1     ,     0     ,
    14.00     ,     0     ,     0     ,
    22.00     ,     1     ,     0     ,
    51.00     ,     1     ,     0     ,
    18.00     ,     1     ,     0     ,
    45.00     ,     0     ,     1     ,
    28.00     ,     1     ,     0     ,
    21.00     ,     1     ,     0     ,
    27.00     ,     1     ,     0     ,
    36.00     ,     1     ,     0     ,
    27.00     ,     1     ,     0     ,
    15.00     ,     0     ,     1     ,
    27.00     ,     1     ,     0     ,
    26.00     ,     1     ,     0     ,
    22.00     ,     1     ,     0     ,
    24.00     ,     1     ,     0     ,
    29.00     ,     1     ,     0     ;
    // std::cout<<testdata<<std::endl;
    
    
    /*
    std::vector<double> weights;
    double l = testdata.rows();
    double init_weight = 1/l;
    for (int i = 0; i < l; i++) {
        weights.push_back(init_weight);
    }
    std::vector<bool> results=univLogReg(weights, testdata, 1);
     */
    std::vector<bool> results = adaboost(testdata);
    std::cout << "Results: " << std::endl;
    for (int i = 0; i < 11; i++) {
        std::cout << results[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}

