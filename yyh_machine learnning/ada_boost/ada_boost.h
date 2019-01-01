#pragma once

#include <vector>

#define MAX_WEAK_CLASSIFIERS 30 //最大弱分类器的个数

namespace ada_boost{

//弱分类器
struct WeakClassifier
{
  int attr_idx;//哪个属性上的弱分类器/哪个维度
  //分类阈值大小
  //classify_orientation =0时，大于分类阈值为1，小于分类阈值为0，classify_orientation =1时相反
  float classify_val;
  int classify_orientation;//方向和classify_val结合使用
  float weight;//分类器权重
  float error_rate = FLT_MAX;//误差率
  //使用当前分类器计算出来的lable值，训练的时候临时用一下，训练之后就没用了
  std::vector<float> predict_lables;
};

class AdaBoost{
public:
  //训练，train_data最后一列为标签
  bool Train(const std::vector<std::vector<float>>& train_data,int row_size, int column_size);
  //分类预测
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
  //弱分类器，最后一列为弱分类器权重
  std::vector<WeakClassifier> weak_classifier_;
};

}
