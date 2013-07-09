#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
namespace ublas = boost::numeric::ublas;
using namespace std;
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
    // y: +1,0
    double update(const ublas::vector<double>& x, double pweight, double nweight, double L2,  double iter)
    {
        // update each weight
        double zi = classify(x);//predict
        for(size_t k = 0; k < mWeightNew.size(); ++k)
        {
            //mWeightNew(k) = mWeightOld(k) + sLearningRate*(pweight/(pweight+nweight)-zi)*x[k] ;
            mWeightNew(k) = mWeightOld(k) + zi*(pweight/(pweight+nweight)-zi)*x[k] ;
        }
        double dist = norm(mWeightNew, mWeightOld);
        mWeightOld.swap(mWeightNew);
        return dist;
    }
    //http://www.icml-2011.org/papers/231_icmlpaper.pdf
    void scg(const ublas::vector<double>& x, double y, double weight, double L2 = 0.1)
    {
        double z_i = inner_prod(mWeightOld, x);
        double sig= y*sigmoid(-y*z_i);
        double beta = 10;
        //for(size_t k = 0; k < mWeightNew.size(); ++k)
        {
            size_t k = random()%mWeightNew.size();
            double gradient = x[k] * sig;
            //mWeightNew(k) =  (1-sLearningRate * L2) *mWeightOld(k) + sLearningRate * gradient;
            double delta = max(-x[k], gradient/beta);
            mWeightNew(k) = mWeightOld(k) + delta;
        }
        mWeightOld.swap(mWeightNew);
    }
    void init(int dim)
    {
        mWeightOld.resize(dim, 0);
        mWeightNew.resize(dim, 0);
    }
    void save(std::ostream& os)
    {
        for (unsigned int i = 0; i < mWeightNew.size(); ++i)
            os <<  mWeightNew[i] << " " ;
        os << endl;
    }
private:
    ublas::vector<double> mWeightOld;
    ublas::vector<double> mWeightNew;
    static const double sConvergenceRate = 0.0001;
    static const double sLearningRate = 0.0001;
};
int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        cout << "Usage: " << argv[0] << " data_file" << endl;
        return -1;
    }
    LR model;
    ifstream ifs(argv[1]);
    string line;
    getline(ifs, line);
    vector<string> strs;
    boost::split(strs, line, boost::is_any_of(","));
    model.init(int(strs.size()) - 1);//add bias feature
    while(getline(ifs, line))
    {
        boost::split(strs, line, boost::is_any_of(","));
        double pos = boost::lexical_cast<double>(strs[0]);
        double neg = boost::lexical_cast<double>(strs[1]);
        ublas::vector<double> x(strs.size() - 1);
        unsigned int i = 0;
        for (; i < strs.size()-2; ++i)
        {
            x(i) = (boost::lexical_cast<double>(strs[i+2]));
        }
        x(i) = 1;
        model.update(x, pos, neg, 0.2, i);
        //model.scg(x, pos, neg, 0.2);
        static int srand = 1;
        if (srand++ %1000 == 0)
        model.save(std::cout);
    }
    return 0;
}
