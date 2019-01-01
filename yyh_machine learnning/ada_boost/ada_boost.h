#pragma once

#include <vector>

#define MAX_WEAK_CLASSIFIERS 30 //������������ĸ���

namespace ada_boost{

//��������
struct WeakClassifier
{
  int attr_idx;//�ĸ������ϵ���������/�ĸ�ά��
  //������ֵ��С
  //classify_orientation =0ʱ�����ڷ�����ֵΪ1��С�ڷ�����ֵΪ0��classify_orientation =1ʱ�෴
  float classify_val;
  int classify_orientation;//�����classify_val���ʹ��
  float weight;//������Ȩ��
  float error_rate = FLT_MAX;//�����
  //ʹ�õ�ǰ���������������lableֵ��ѵ����ʱ����ʱ��һ�£�ѵ��֮���û����
  std::vector<float> predict_lables;
};

class AdaBoost{
public:
  //ѵ����train_data���һ��Ϊ��ǩ
  bool Train(const std::vector<std::vector<float>>& train_data,int row_size, int column_size);
  //����Ԥ��
  bool Classify(std::vector<int>&result, const std::vector<std::vector<float>>& test_data);
private:
  bool TrainIter(
    const std::vector<std::vector<float>>& train_data, 
    std::vector<float>& row_weight, std::vector<float>& weak_classifier_add, WeakClassifier& weak_class);

  WeakClassifier GetBestWeakClassifier(
    const std::vector<std::vector<float>>& train_data,std::vector<float>& row_weight);

  std::vector<float> ClassifierFunction(
    const std::vector<std::vector<float>>& train_data, WeakClassifier& weak_class);

  bool CheckData(const std::vector<std::vector<float>>& train_data, int row_size, int column_size);
  //�������������һ��Ϊ��������Ȩ��
  std::vector<WeakClassifier> weak_classifier_;
};

}
