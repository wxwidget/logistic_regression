#include "data.h"
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
using namespace boost;
double** dmatrix(int row, int col) {
    double** x = new double*[row];
    for(int i = 0; i < row; ++i) {
        x[i] = new double[col];
        memset(x[i], 0, sizeof(double)*col);
    }
    return x;
}
void free_matrix(double**x, int row) {
    for(int i = 0; i < row; ++i) {
        delete[] x[i];
    }
    delete[] x;
}
double* dvector(int col) {
    double* x = new double[col];
    memset(x, 0, sizeof(double)*col);
    return x;
}
void free_vector(double* x){
    delete []x;
}
bool ends_with(const string& str, const string& a) {
    int n = a.size();
    int s = str.size() - a.size();
    int i = 0;
    while(str[s+i] == a[i] && i < n) {
        i++;
    }
    return i == n;
}
void csv_read(const char* in_string, double*x) {
    const char* pos = in_string;
    int i = 0;
    for(; pos - 1 != NULL && pos[0] != '\0' && pos[0] != '#';
            pos = strchr(pos, ',') + 1) {
        float value = atof(pos);
        x[i++] = value;
    }
}
void svm_read_line(const string& line, double* x, double y) {
    //libsvm format
    //<class/target> [<attribute number>:<attribute value>]*
    //csv format
    //target[,features_values]
    vector<string> strs;
    split(strs, line, is_any_of("\t ,"));
    double target = 0;
    if(strs[0] == "+1") target = 1;
    else if(strs[0] == "1") target = 1;
    for(unsigned int i = 0; i < strs.size(); ++i) {
        vector<string> kvs;
        split(kvs, strs[i], is_any_of(":"));
        if(kvs.size() == 2) { //id:value for libsvm
            int index = lexical_cast<int>(kvs[0]);
            x[index] = lexical_cast<double>(kvs[1]);
        } else if(kvs.size() == 1) { //value for csv
            x[i] =  lexical_cast<double>(kvs[0]);
        }
    }
    y = target;
}
void svm_load(const string& filename, double** x, double* y) {
    int cur_row = 0;
    ifstream ifs(filename.c_str());
    string line;
    while(getline(ifs, line)) {
        svm_read_line(line, x[cur_row], y[cur_row]);
        cur_row++;
    }
}
void csv_load_feature(const char* train_feature, double**x) {
    int cur_row = 0;
    ifstream ifs(train_feature);
    string line;
    while(getline(ifs, line)) {
        csv_read(line.c_str(), x[cur_row]);
        cur_row++;
    }
}
void load_target(const char* train_target, double* y) {
    int cur_row = 0;
    ifstream ifs(train_target);//target 0 or 1
    double temp = 0;
    while(ifs >> temp) {
        y[cur_row++] = temp;
    }
}
