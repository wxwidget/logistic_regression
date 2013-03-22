#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
using namespace std;
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
    void train(ublas::matrix<double>& x, ublas::vector<double>& y, double regularization = 0.1)
    {
        int max_iters = 2000;
        mWeightOld.resize(x.size2(), 0);
        mWeightNew.resize(x.size2(), 0);
        for(int iter = 0; iter < max_iters; ++iter)
        {
            // update each weight
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
                    gradient = y(i) * x(i, k) * sigmoid(-y(i) * z_i);
                }
                mWeightNew(k) = mWeightOld(k) + sLearningRate * gradient -
                                sLearningRate * regularization * mWeightOld(k);
            }
            double dist = norm(mWeightNew, mWeightOld);
            if(dist < sConvergenceRate)
            {
                break;
            }
            mWeightOld.swap(mWeightNew);
        }
    }
private:
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
    const int record_num = 270;
    const int dim_num = 13 + 1;
    ublas::vector<double> y(record_num);
    ublas::matrix<double> x(record_num, dim_num);
    //dataset loader(record_num, dim_num);
    //loader.load_file(argv[1], y, x);
    LR model;
    model.train(x, y);
    return 0;
}
