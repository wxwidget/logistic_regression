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
    void load(istream& in)
    {
        string line;
        getline(in, line);
        vector<string> strs;
        boost::split(strs, line, boost::is_any_of("\t "));
        for (unsigned int i = 0; i < strs.size(); ++i)
        {
            vector<string> kv;
            boost::split(kv, strs[i], boost::is_any_of(":"));
            if (kv.size() == 2)
            weight[kv[0]] = boost::lexical_cast<double>(kv[1]);
        }
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
    if(argc != 3)
    {
        cout << "Usage: " << argv[0] << " <model_file> <test_file>" << endl;
        return -1;
    }
    LR model(1);
    ifstream mfile(argv[1]);
    model.load(mfile);
    ifstream ifs(argv[2]);
    unordered_map<string, double> features;
    string line;
    while(getline(ifs, line))
    {
        //libsvm format 
        //<class/target> [<attribute number>:<attribute value>]*
        vector<string> strs;
        features.clear();
        boost::split(strs, line, boost::is_any_of("\t "));
        double target =  boost::lexical_cast<double>(strs[0]);
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
        double t = model.classify(features);
        cout << t  << "\t" << target << endl;
    }
    return 0;
}
