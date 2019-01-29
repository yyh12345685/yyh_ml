#pragma once
#include <vector>
#include <string>
#include <map>

namespace naive_bayes{

struct Doc {
  std::vector<std::vector<std::string> >doc_words;
  std::vector<int> classes;
  std::map<std::string, int>word_of_index;//value index start with 1
  std::vector<std::vector<int> > doc_matrix;//文档矩阵，每行一个向量
};

class NaiveBayes{
public:
  bool InitDoc(
    const std::vector<std::vector<std::string> >&doc_words,
    const std::vector<int>& classes);

  void Trainning();
  int Classify(const std::vector<std::string>& doc);
protected:
  bool InitDocMatrix();
private:
  Doc doc_;
  //只支持二分类
  std::vector<float> p0_vect_; //负样本时，每一列的概率即：p(xi|y=0)
  std::vector<float> p1_vect_; //正样本，每一列的概率即：p(xi|y=1)
  float p_abusive_;//负样本概率
};

}
