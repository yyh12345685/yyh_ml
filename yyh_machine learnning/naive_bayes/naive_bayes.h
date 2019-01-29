#pragma once
#include <vector>
#include <string>
#include <map>

namespace naive_bayes{

struct Doc {
  std::vector<std::vector<std::string> >doc_words;
  std::vector<int> classes;
  std::map<std::string, int>word_of_index;//value index start with 1
  std::vector<std::vector<int> > doc_matrix;//�ĵ�����ÿ��һ������
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
  //ֻ֧�ֶ�����
  std::vector<float> p0_vect_; //������ʱ��ÿһ�еĸ��ʼ���p(xi|y=0)
  std::vector<float> p1_vect_; //��������ÿһ�еĸ��ʼ���p(xi|y=1)
  float p_abusive_;//����������
};

}
