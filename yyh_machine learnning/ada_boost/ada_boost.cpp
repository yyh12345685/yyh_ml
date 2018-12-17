#include <algorithm>
#include <iostream>
#include "ada_boost.h"
#include "common.h"

bool AdaBoost::Train(
  const std::vector<std::vector<float>>& train_data,int row_size,int column_size){
  if (!CheckData(train_data,row_size,column_size)){
    return false;
  }

  std::vector<float> row_weight(row_size);//每个样例的权重
  for (auto& weight:row_weight){
    weight = common::GetReservedDecimalNums(1.0 / row_size, 6);
  }

  std::vector<float> weak_classifier_add(row_size);  //所有弱分类器的预测加权结果
  memset(&weak_classifier_add[0], '\0', sizeof(float)*row_size);

  for (int idx = 0; idx < MAX_WEAK_CLASSIFIERS;idx++) {
    WeakClassifier weak_classifier;
    if (TrainIter(train_data,row_weight, weak_classifier_add, weak_classifier)){
      weak_classifier_.push_back(weak_classifier);
      break;
    }
    weak_classifier_.push_back(weak_classifier);
  }
  std::cout << "weak_classifier size:" << weak_classifier_.size() << std::endl;
  return true;
}

int Sign(float a)
{
  if (a >= 0) 
    return 1;
  else 
    return -1;
}

bool AdaBoost::TrainIter(
  const std::vector<std::vector<float>>& train_data,
  std::vector<float>& row_weight, std::vector<float>& weak_classifier_add, WeakClassifier& weak_class) {
  weak_class = GetBestWeakClassifier(train_data, row_weight);
  weak_class.weight = 0.5 * log((1.0 - weak_class.error_rate) / std::max(weak_class.error_rate, FLT_MIN));
  //为下一次迭代，计算每个样例的权重D
  float sum = 0;
  for (size_t row_idx = 0; row_idx < train_data.size(); row_idx++)
  {
    float expon = -1.0 * weak_class.weight *
      weak_class.predict_lables[row_idx] * train_data[row_idx][train_data[row_idx].size() - 1];
    row_weight[row_idx] *= exp(expon);
    sum += row_weight[row_idx];
  }
  int error_count = 0;//预测错误的条数
  for (size_t row_idx = 0; row_idx < train_data.size(); row_idx++)
  {
    row_weight[row_idx] /= sum;
    weak_classifier_add[row_idx] += weak_class.weight * weak_class.predict_lables[row_idx];
    if (Sign(weak_classifier_add[row_idx]) !=
      train_data[row_idx][train_data[row_idx].size() - 1])
      error_count++;
  }
  if (0 == error_count){
    return true;
  }
  return false;
}

WeakClassifier AdaBoost::GetBestWeakClassifier(
  const std::vector<std::vector<float>>& train_data,std::vector<float>& row_weight){
  int row_size = train_data.size();
  int col_size = train_data[0].size();
  int num_steps = 10;//默认分为10份
  WeakClassifier min_weak_classifier;
  //遍历所有特征,最后一列为标签列
  for (int col_idx = 0; col_idx < col_size-1; col_idx++)
  {//按照列来遍历，一列才是一个维度
    float col_min = train_data[0][col_idx];
    float col_max = train_data[0][col_idx];
    //求当前特征上的最小值、最大值
    for (int row_idx = 1; row_idx < row_size -1; row_idx++)
    {
      if (train_data[row_idx][col_idx] < col_min)
        col_min = train_data[row_idx][col_idx];
      if (train_data[row_idx][col_idx] > col_max)
        col_max = train_data[row_idx][col_idx];
    }
    float step_size = (col_max - col_min) / num_steps;  //步长
    for (int step_idx = 0; step_idx < num_steps; step_idx++)
    {
      float classify_val = col_min + step_idx * step_size;
      for (int classify_orientation = 0; classify_orientation < 2; classify_orientation++)  
      {//在大于、小于之间切换不等式
        WeakClassifier weak_classifier;
        weak_classifier.attr_idx = col_idx;
        weak_classifier.classify_orientation = classify_orientation;
        weak_classifier.classify_val = common::GetReservedDecimalNums(classify_val, 6);
        std::vector<float> predict_vals = ClassifierFunction(train_data, weak_classifier);  //预测结果
        float weight_error = 0.0;
        for (size_t row_idx = 0; row_idx < train_data.size()&& train_data[row_idx].size(); row_idx++)
        {
          if (predict_vals[row_idx] != train_data[row_idx][train_data[row_idx].size()-1])
          {
            weight_error += row_weight[row_idx];  //计算加权错误率
          }
        }
        weight_error = common::GetReservedDecimalNums(weight_error, 6);
        if (weight_error < min_weak_classifier.error_rate)
        {
          min_weak_classifier.error_rate = weight_error;
          min_weak_classifier.predict_lables = predict_vals;
          min_weak_classifier.attr_idx = col_idx;
          min_weak_classifier.classify_orientation = classify_orientation;
          min_weak_classifier.classify_val = common::GetReservedDecimalNums(classify_val, 6);
        }
      }
    }
  }
  return min_weak_classifier;
}

std::vector<float> AdaBoost::ClassifierFunction(
  const std::vector<std::vector<float>>& train_data, WeakClassifier& weak_class){
  int row_size = train_data.size();
  std::vector<float> ret_label(row_size);
  if (weak_class.classify_orientation == 0){
    for (int idx = 0; idx < row_size; idx++)
      if (train_data[idx][weak_class.attr_idx] <= weak_class.classify_val)
        ret_label[idx] = -1;
      else
        ret_label[idx] = 1;
  } else{
    for (int idx = 0; idx < row_size; idx++)
      if (train_data[idx][weak_class.attr_idx] > weak_class.classify_val)
        ret_label[idx] = -1;
      else
        ret_label[idx] = 1;
  }
  return ret_label;
}

bool AdaBoost::CheckData(
  const std::vector<std::vector<float>>& train_data, int row_size, int column_size){
  if (row_size != train_data.size()){
    return false;
  }

  for (const auto& vec:train_data){
    if (vec.size()!=column_size){
      return false;
    }
  }

  return true;
}

bool AdaBoost::Classify(
  std::vector<int>&result, const std::vector<std::vector<float>>& test_data){
  if (test_data.size()==0 || test_data[0].size() == 0){
    return false;
  }

  if (!CheckData(test_data, test_data.size(), test_data[0].size())){
    return false;
  }
  std::vector<float>result_float;
  result_float.resize(test_data.size());
  memset(&result_float[0], '\0', sizeof(float)*test_data.size());

  for (int idx = 0; idx < weak_classifier_.size(); idx++){
    std::vector<float> predict_lables = ClassifierFunction(test_data,weak_classifier_[idx]);
    for (int row_idx = 0; row_idx < test_data.size(); row_idx++)
      result_float[row_idx] += weak_classifier_[idx].weight * predict_lables[row_idx];
  }
  result.resize(result_float.size());
  for (int idx = 0; idx < result_float.size(); idx++)
    result[idx] = Sign(result_float[idx]);
  return true;
}
