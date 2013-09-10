#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
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
    void train(ublas::matrix<double>& x, ublas::vector<double>& y, double regularization = 0.5)
    {
        int max_iters = 40;
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
                    gradient += y(i) * x(i, k) * sigmoid(-y(i) * z_i);
                }
                mWeightNew(k) = mWeightOld(k) + sLearningRate * gradient - sLearningRate * regularization * mWeightOld(k);
            }
            double dist = norm(mWeightNew, mWeightOld);
            if(dist < sConvergenceRate)
            {
                break;
            }
            mWeightOld.swap(mWeightNew);
        }
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
    static const double sLearningRate = 0.0005;
};
int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        cout << "Usage: " << argv[0] << " <data_file> <instance_count> <feature_count>" << endl
             << "\t data_file: the training date\n"
             << "\t instance_count: the trainning instance count in data_file\n"
             << "\t feature_count: the feature count in data_file, including bias\n";
        return -1;
    }
    ifstream ifs(argv[1]);
    string line;
    
    int row = boost::lexical_cast<int>(argv[2]);
    int col = boost::lexical_cast<int>(argv[3]);
    ublas::matrix<double> x(row, col);
    ublas::vector<double> y(row);
    int cur_row = 0;
    //cerr << "rows:" << row << " col:" << col << endl;
    while(getline(ifs, line))
    {
        //libsvm format 
        //<class/target> [<attribute number>:<attribute value>]*
        vector<string> strs;
        boost::split(strs, line, boost::is_any_of("\t "),boost::algorithm::token_compress_on);
        double target = -1;//boost::lexical_cast<double>(strs[0]);
        if (strs[0] == "+1") target = 1;
        for (unsigned int i = 1; i < strs.size(); ++i)
        {
            vector<string> kvs;
            boost::split(kvs, strs[i], boost::is_any_of(":"),boost::algorithm::token_compress_on);
            if (kvs.size()==2) 
            {
                int index = boost::lexical_cast<int>(kvs[0]);
                x(cur_row, index) = boost::lexical_cast<double>(kvs[1]);
            }
            else if (kvs.size() == 1 && !kvs[0].empty())
            {
                int index = boost::lexical_cast<int>(kvs[0]);
                x(cur_row, index) = 1;
            }
        }
        x(cur_row,0) = 1;//bias
        y(cur_row) = target;
        cur_row++;
    }
    LR model;
    model.train(x, y);
    model.save(std::cout);
    return 0;
}
