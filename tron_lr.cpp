//http://people.csail.mit.edu/jrennie/writing/lr.pdf  y=+1,-1
#include <iostream>
#include <cmath>
#include <memory>
#include <fstream>
#include "data.h"
using namespace std;
typedef struct instance_t{
    double** x;
    double* y;
    int row;
    int col;
    ~instance_t(){
        free_matrix(x, row);
        free_vector(y);
    }
}instance;

template<typename T1, typename T2>
static float dot(const T1* x, const T2* y, int n) {
    float r = 0;
    for(int i = 0; i < n; ++i) {
        r += x[i] * y[i];
    }
    return r;
}

template<typename T1, typename T2>
static int binary(const T1* x, const T2* y, int n) {
    return dot(x,y,n) > 0 ? 1: 0;
}
template <typename FType>
static  FType sigmoid(FType x) {
    double e = 2.718281828;
    if (x >= 10){
        return 1.0 / (1.0 + pow(e, -10));
    }else if (x <= -10){
        return 1.0 / (1.0 + pow(e, 10));
    }
    return 1.0 / (1.0 + pow(e, -x));
}
template<typename T1, typename T2>
static float predict(const T1* x, const T2* weight, int n) {
    return sigmoid(dot(x,weight,n));
}
static float evaluate(void *ins, const float *w, float *g, const int n, const float step) {
    float e = 2.718281828;
    float fx = 0.0;
    instance* data = (instance*)ins; 
    double** x = data->x;
    double* y = data->y;
    for(int i = 0; i < n; ++i){//for variable
        for (int j = 0;j < data->row; ++j){ //for instance
            float p = predict(x[j], w, n);//predict instance
            g[i] += (p - y[j])*x[j][i];
        }
        g[i] = g[i]/data->row + w[i];
    }
    for (int j = 0;j < data->row; ++j){//for instance
        float wx = dot(x[j], w,n);
        if (y[j] < 0.5){//y[i] == 0
            fx +=  log(1+pow(e, wx));
        }
        else{
            fx +=  log(1+pow(e, -wx));//log loss
        }
    }
    fx = fx/data->row + dot(w,w,n)/2;
    return fx;
}
int main(int argc, char* argv[]) {
    if(argc < 4) {
        cerr << "Usage: " << argv[0] << " <train_feature> <train_target> <rows> <cols> [test]" << endl
             << "\t data_file: the training date\n";
        return -1;
    }
    const char* feature = argv[1];
    const char* target = argv[2];
    int row = atoi(argv[3]);
    int col = atoi(argv[4]);
    instance ins;
    ins.x = dmatrix(row, col);
    ins.y = dvector(row);
    ins.row = row;
    ins.col = col;
    load_feature(feature, ins.x);
    load_target(target, ins.y);

    float fx;
    //lbfgs(col, w, &fx, evaluate, progress, &ins, &param);

    double** confuse = dmatrix(2, 2);
    for(int i = 0; i < row; ++i) {
        int pred = binary(ins.x[i], w, col);
        int label = (int)ins.y[i];
        confuse[label][pred]++;
    }
    cout << "model=>";
    for(int i = 0; i < col; ++i){
        cout << i <<":" << w[i] << " ";
    }
    cout << "L\t0\t1\tprecision\tsupprt<-prdict\n";
    double label0 = confuse[0][0] + confuse[0][1];
    double label1 = confuse[1][0] + confuse[1][1];
    cout << "0\t" << confuse[0][0] << "\t" << confuse[0][1] << "\t" << confuse[0][0]/label0 << "\t" << label0 << endl;
    cout << "1\t" << confuse[1][0] << "\t" << confuse[1][1] << "\t" << confuse[1][1]/label1 << "\t" << label1 << endl;
    return 0;
}
