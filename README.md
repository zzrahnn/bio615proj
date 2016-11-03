# bio615proj


Proposal:
Description of original ADABOOST algorithm: 
We will be implementing a specific case of ADABOOST, an ensemble classification alogorithm. Specifically, we will be implementing a version with “soft margins,” which allows for more classification errors and is therefore less prone to overfitting on noisey data with many outliers. 
The paper we will be following is: RATSCH, G. “Soft Margins for AdaBoost.” Machine Learning, 42, 287–320, 2001. 
Our data will consist of a single feature and a binary outcome variable. Our predictor will be (continuous, and we will use logistic regression as the weak classifier) (binary, and we will use Naive Bayes as the weak classifier). 
A general version of the algorithm is as follows:
Input the training data and choose initial weights for each data pair; for initializing, just weight each data pair the same (must add to 1)
Train the classifier with respect to the weighted sample set and obtain hypothesis classifications (map x to y).
We can choose which base classifier to use; options include logistic regression, knn, naiive bayes, decision trees; we will choose one and stick with it. We should try to pick the easiest one, as recommended by the Professor. 
Calculate the training error (the proportion of incorrect classifications), given the weight of each data pair, which will always be less than 0.5.
Calculate the log odds of the training error ( (1- error) / error)
Update the weights for each data pair, using the odds of the training error. If the original pair was classified incorrectly, its weight goes up; if the pair was classified correctly, the weight goes down. 
Update until there is no more error;  break if the error is larger than 0.5 (your base classifier is not working as specified). 
A general version of the algorithm is as follows: (soft margin)
Input the training data and choose initial weights for each data pair; for initializing, just weight each data pair the same (must add to 1)
Train the classifier with respect to the weighted sample set and obtain hypothesis classifications (map x to y).
We can choose which base classifier to use; options include logistic regression, knn, naiive bayes, decision trees; we will choose one and stick with it. 
Calculate the training error (the proportion of incorrect classifications), given the weight of each data pair, which will always be less than 0.5.
Calculate the log odds of the training error ( (1- error) / error)
Update the weights for each data pair, using the odds of the training error. If the original pair was classified incorrectly, its weight goes down; if the pair was classified correctly, the weight goes up. 
Update until there is no more error;  break if the error is larger than 0.5 (your base classifier is not working as specified). 
