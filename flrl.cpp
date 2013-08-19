/*
 * ref: 
 *
 * http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/pubs/archive/41159.pdf
 * http://jmlr.org/proceedings/papers/v15/mcmahan11b/mcmahan11b.pdf
 */
#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <tr1/unordered_map>
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;
using namespace std::tr1;
class LR
{
public:
    LR(double a, double b = 1): alpha(a), beta(b)
    {
    }
    double sigmoid(double x)
    {
        double e = 2.718281828;
        return 1.0 / (1.0 + pow(e, -x));
    }
    double classify(const unordered_map<string, double>& x)
    {
        unordered_map<string, double>::const_iterator it = x.begin();
        double y =  0;
        for (; it != x.end(); ++it)
        {
            y += weight[it->first] * it->second;
        }
        return sigmoid(y);
    }
    static int sgn(double x)
    {
        if (x>0) return(1); 
        if (x<0) return(-1);
        return(0); 
    }
    // y: +1,0
    double update(const unordered_map<string, double>& x, double target, double L1, double L2,  double iter)
    {
        unordered_map<string,double>::const_iterator it = x.begin();
        for(; it != x.end(); ++it)
        {
            const string& name = it->first;
            if (fabs(zi[name]) <= L1)
            {
                weight[name] = 0;
            }
            else
            {
                weight[name] = -(zi[name] - sgn(zi[name])*L1)/((beta + sqrt(ni[name]))/alpha + L2);
            }
        }
        double pt = classify(x); 
        it = x.begin();
        for(; it != x.end(); ++it)
        {
           const string& name = it->first;
           double value = it->second;
           double gi = (pt - target) * value;
           double thi = (sqrt(ni[name] + gi * gi) - sqrt(ni[name]))/alpha;
           zi[name] += gi - thi * weight[name];
           ni[name] += gi * gi;
        }
        return 0;
    }
    void save(ostream& osr)
    {
        unordered_map<string, double>::iterator it = weight.begin();
        for(; it != weight.end(); ++it)
        {
            osr << it->first << ":" << it->second << " ";
        }
        osr << endl;
    }
private:
    unordered_map<string,double> weight;
    unordered_map<string,double> zi;
    unordered_map<string,double> ni;
    double alpha;
    double beta;
};
int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        cout << "Usage: " << argv[0] << " data_file" << endl;
        return -1;
    }
    LR model(0.5);
    ifstream ifs(argv[1]);
    unordered_map<string, double> features;
    string line;
    int iterator = 1;
    while(getline(ifs, line))
    {
        //libsvm format 
        //<class/target> [<attribute number>:<attribute value>]*
        features.clear();
        vector<string> strs;
        boost::split(strs, line, boost::is_any_of("\t "));
        double target = 0;//boost::lexical_cast<double>(strs[0]);
        if (strs[0] == "+1") target = 1;
        //double target = boost::lexical_cast<double>(strs[0]);
        for (unsigned int i = 1; i < strs.size(); ++i)
        {
            vector<string> kvs;
            boost::split(kvs, strs[i], boost::is_any_of(":"));
            if (kvs.size()==2) 
            {
                features[kvs[0]] = boost::lexical_cast<double>(kvs[1]);
            }
            else if (kvs.size() == 1)
            {
                features[kvs[0]] = 1;
            }
        }
        features["b"] = 1;
        model.update(features, target, 0.5, 0.5, iterator++);
    }
    model.save(std::cout);
    return 0;
}
