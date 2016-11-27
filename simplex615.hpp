
#define ZEPS 1e-10
#include <Eigen/Dense>

template <class F> // F is a function object
class simplex615 { // contains (dim+1) points of size (dim)
    
protected:
    std::vector<std::vector<double> > X; // (dim+1)*dim matrix
    std::vector<double> Y; // (dim+1) vector
    std::vector<double> midPoint; // variables for update
    std::vector<double> thruLine; // variables for update
    int dim, idxLo, idxHi, idxNextHi; // dimension, min, max, 2ndmax values
    void evaluateFunction(F& foo); // evaluate function value at each point
    void evaluateExtremes(); // determine the min, max, 2ndmax
    void prepareUpdate(); // calculate midPoint, thruLine
    bool updateSimplex(F& foo, double scale); // for reflection/expansion..
    void contractSimplex(F& foo); // for multiple contraction
    static int check_tol(double fmax, double fmin, double ftol); // check tolerance
    
    // attempting to give data to simplex
    Eigen::MatrixXd &data;
    int column;
    std::vector<double> weights;
    
    
public:
    simplex615(double* p, int d, Eigen::MatrixXd &data, int column, std::vector<double> weights); // constructor with initial points
    void amoeba(F& foo, double tol); // main function for optimization
    std::vector<double>& xmin(); // optimal x value
    double ymin(); // optimal y value
};

template <class F> simplex615<F>::simplex615(double* p, int d, Eigen::MatrixXd &dat, int col, std::vector<double> wts) : dim(d), data(dat) { // set dimension
    // Determine the space required
    X.resize(dim+1); // X is vector-of-vector, like 2-D array
    Y.resize(dim+1); // Y is function value at each simplex point
    midPoint.resize(dim); thruLine.resize(dim);
    for(int i=0; i < dim+1; ++i) { X[i].resize(dim); // allocate the size of content in the 2-D array
    } // Initially, make every point in the simplex identical
    for(int i=0; i < dim+1; ++i)
        for(int j=0; j < dim; ++j) X[i][j] = p[j]; // set each simple point to the starting point // then increase each dimension by one unit except for the last point
    for(int i=0; i < dim; ++i) X[i][i] += 1.; // this will generate a simplex
    // attempting to give data to simplex
    // data = dat;
    column = col;
    weights = wts;
}

template <class F>
void simplex615<F>::evaluateFunction(F& foo) {
    for(int i=0; i < dim+1; ++i) {
        Y[i] = foo(X[i], data, column, weights); // foo is a function object, which will be visited later
    }
}

template <class F>
void simplex615<F>::evaluateExtremes() {
    if ( Y[0] > Y[1] ) { // compare the first two points
        idxHi = 0; idxLo = idxNextHi = 1;
    }
    else {
        idxHi = 1; idxLo = idxNextHi = 0;
    }
    // for each of the next points
    for(int i=2; i < dim+1; ++i) {
        if ( Y[i] <= Y[idxLo] ) // update the best point if lower
            idxLo = i;
        else if ( Y[i] > Y[idxHi] ) { // update the worst point if higher
            idxNextHi = idxHi; idxHi = i;
        }
        else if ( Y[i] > Y[idxNextHi] ) // update also if it is the 2nd-worst point
            idxNextHi = i;
    }
}

template <class F> void simplex615<F>::prepareUpdate() {
    for(int j=0; j < dim; ++j) {
        midPoint[j] = 0; // average of all points but the worst point
    }
    for(int i=0; i < dim+1; ++i) {
        if ( i != idxHi ) { // exclude the worst point
            for(int j=0; j < dim; ++j) {
                midPoint[j] += X[i][j]; }
        }
    }
    for(int j=0; j < dim; ++j) { midPoint[j] /= dim;
        thruLine[j] = X[idxHi][j] - midPoint[j]; // direction for optimization
    }
}


template <class F>
bool simplex615<F>::updateSimplex(F& foo, double scale) {
    std::vector<double> nextPoint; // next point to evaluate
    nextPoint.resize(dim);
    for(int i=0; i < dim; ++i) {
        nextPoint[i] = midPoint[i] + scale * thruLine[i];
    }
    double fNext = foo(nextPoint,data,column,weights);
    if ( fNext < Y[idxHi] ) { // update only maximum values (if possible)
        for(int i=0; i < dim; ++i) { // because the order can be changed with
            X[idxHi][i] = nextPoint[i]; // evaluateExtremes() later
        }
        Y[idxHi] = fNext;
        return true;
    }
    else {
        return false; // never mind if worse than the worst
    }
}

template <class F> void simplex615<F>::contractSimplex(F& foo) {
    for(int i=0; i < dim+1; ++i) { if ( i != idxLo ) { // except for the minimum point
        for(int j=0; j < dim; ++j) { X[i][j] = 0.5*( X[idxLo][j] + X[i][j] ); // move the point towards minimum
        }
        Y[i] = foo(X[i], data, column, weights); // re-evaluate the function
    }
    }
}


template <class F>
void simplex615<F>::amoeba(F& foo, double tol) {
    evaluateFunction(foo); // evaluate the function at the initial points
    while(true) {
        evaluateExtremes(); // determine three important points
        prepareUpdate(); // determine direction for optimization
        if ( check_tol(Y[idxHi],Y[idxLo],tol) ) break; // check convergence
        updateSimplex(foo, -1.0); // reflection
        if ( Y[idxHi] < Y[idxLo] ) {
            updateSimplex(foo, -2.0); // expansion
        }
        else if ( Y[idxHi] >= Y[idxNextHi] ) {
            if ( !updateSimplex(foo, 0.5) ) { // 1-d contraction
                contractSimplex(foo); // multiple contractions
            }
        }
    }
}
template <class F> int simplex615<F>::check_tol(double fmax, double fmin, double ftol) {
    // calculate the difference
    double delta = fabs(fmax - fmin); // calculate the relative tolerance
    double accuracy = (fabs(fmax) + fabs(fmin)) * ftol; // check if difference is within tolerance
    return (delta < (accuracy + ZEPS));
}

template <class F>
std::vector<double>& simplex615<F>::xmin(){
    return X[idxLo];
}

template <class F>
double simplex615<F>::ymin(){
    return Y[idxLo];
}







