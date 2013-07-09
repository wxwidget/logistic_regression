#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
using namespace std;
using namespace boost;
namespace ublas = boost::numeric::ublas;
class LR
{
public:
    double norm(const ublas::vector<double>& v1, const ublas::vector<double>& v2)
    {
        assert(v1.size() == v2.size());
        double sum = 0;
        for(size_t i = 0; i < v1.size(); ++i)
        {
            double minus = v1(i) - v2(i);
            double r = minus * minus;
            sum += r;
        }
        return sqrt(sum);
    }
    double sigmoid(double x)
    {
        double e = 2.718281828;
        return 1.0 / (1.0 + pow(e, -x));
    }
    double classify(const ublas::vector<double>& x)
    {
        double y =  inner_prod(x, mWeightNew);
        return sigmoid(y);
    }
    //http://people.csail.mit.edu/jrennie/writing/lr.pdf
    // y: +1,-1
    void train(ublas::matrix<double>& x, ublas::vector<double>& pos, ublas::vector<double>& neg,double regularization = 0.1)
    {
        int max_iters = 20;
        mWeightOld.resize(x.size2(), 0);
        mWeightNew.resize(x.size2(), 0);
        for(int iter = 0; iter < max_iters; ++iter)
        {
            for(size_t k = 0; k < mWeightNew.size(); ++k)
            {
                double gradient = 0;
                for(size_t i = 0; i < x.size1(); ++i)
                {
                    double z_i = 0;
                    for(size_t j = 0; j < mWeightOld.size(); ++j)
                    {
                        z_i += mWeightOld(j) * x(i, j);
                    }
                    gradient += x(i, k) * sigmoid(-z_i) * pos(i);
                    gradient += -x(i, k) * sigmoid(z_i) * neg(i);
                }
                mWeightNew(k) = mWeightOld(k) + sLearningRate * gradient - sLearningRate * regularization * mWeightOld(k);
            }
            std::cout <<  mWeightNew << std::endl;
            double dist = norm(mWeightNew, mWeightOld);
            if(dist < sConvergenceRate)
            {
                break;
            }
            mWeightOld.swap(mWeightNew);
        }
    }
    ublas::vector<double> mWeightOld;
    ublas::vector<double> mWeightNew;
    static const double sConvergenceRate = 0.0001;
    static const double sLearningRate = 0.00005;
};
int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        cout << "Usage: " << argv[0] << " data_file" << endl;
        return -1;
    }

    ifstream ifs(argv[1]);
    string line;
    getline(ifs, line);
    vector<string> strs;
    boost::split(strs, line, boost::is_any_of(","));
    int rows = 0;
    int cols = strs.size();
    while(getline(ifs, line))
    {
        rows++;
    }
    ublas::matrix<double> x(rows, cols);
    ublas::vector<double> pweight(rows);
    ublas::vector<double> nweight(rows);

    ifs.clear();
    ifs.seekg(0, ios::beg);  
    getline(ifs, line);
    for(int row = 0; row < rows; ++row)
    {
        getline(ifs, line);
        boost::split(strs, line, boost::is_any_of(","));
        double pos = boost::lexical_cast<double>(strs[0]);
        double neg = boost::lexical_cast<double>(strs[1]);
        unsigned int i = 2;
        for (int i = 2; i < strs.size(); ++i)
        {
            x(row, i-2) = lexical_cast<double>(strs[i]);
        }
        x(row,i-2) = 1;
        pweight[row] = pos;
        nweight[row] = neg;
    }
    LR model;
    model.train(x, pweight, nweight);
    std::cout <<  model.mWeightNew << std::endl;
    return 0;
}
