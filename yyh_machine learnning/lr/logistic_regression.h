#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include "common.h"

namespace lr {

template <typename Type>
struct DataNode {
  std::vector<Type> data;
  Type lable;
};

template <typename Type>
class LogisticRegression{
public:
  LogisticRegression(const int& epoch_times){
    epoch_ = epoch_times;
  }
  bool InitData(const std::string& file_name);
  bool TrainBgd();
  bool TrainSgd();
  Type Predict(const std::vector<Type>& input_data);
protected:
  Type GetTotalWeight(const std::vector<Type>& data);
  Type Sigmoid(const Type& res) {
    return 1.0 / (1 + exp((-1)*res));
  }

  Type GetLoss(std::vector<Type>& predict_list);
private:
  std::vector<Type> weight_;//weight_[0]为截距

  std::vector<DataNode<Type> > data_set_;//train data
  int epoch_ = 20;
  Type learning_rate_ = Type(0.1);
};

template<typename Type>
bool LogisticRegression<typename Type>::InitData(
  const std::string& file_name) {
  std::ifstream read_file(file_name);
  if (!read_file.is_open()) {
    std::cout << "Open read file fail,file_path:" << file_name << std::endl;
    return false;
  }
  std::string line_string;
  while (std::getline(read_file, line_string)) {
    std::vector<Type> line_item;
    common::Split(line_item, line_string, '\t', false);
    if (line_item.size() <= 1) {
      continue;
    }
    DataNode<Type> row;
    row.data.emplace_back(1);//用来做截距的相乘，训练需要，测试不需要
    for (size_t idx = 0; idx < line_item.size() - 1; idx++) {
      row.data.emplace_back((line_item[idx]));
    }

    row.lable = line_item[line_item.size() - 1];
    data_set_.emplace_back(row);
  }

  if (data_set_.empty() || data_set_[0].data.empty()) {
    return false;
  }

  weight_.resize(data_set_[0].data.size(), 1.0);
  //srand((int)time(NULL));
  //int random_num = rand();//生成随机整数，做随机种子使用
  //std::default_random_engine generator;
  //generator.seed(random_num);
  ////生成均值为0，方差为1的正态分布
  //std::normal_distribution<float> distribution(0.0, 1.0);
  //for (size_t idx = 0; idx < weight_.size(); idx++) {
  //  weight_[idx] = distribution(generator);
  //}

  return true;
}

template<typename Type>
Type LogisticRegression<typename Type>::GetLoss(std::vector<Type>& predict_list){
  Type loss = 0;
  if (predict_list.size()!= data_set_.size()){
    return loss;
  }
  for (size_t idx = 0; idx < predict_list.size();idx++) {
    loss += -(data_set_[idx].lable * std::log(predict_list[idx])
      + ((1 - data_set_[idx].lable) * std::log(1 - predict_list[idx])));
  }
  loss /=  data_set_.size();
  return loss;
}

template<typename Type>
bool LogisticRegression<typename Type>::TrainBgd(){
  learning_rate_ = 0.1;
  for (int epoch = 0; epoch < epoch_;epoch++) {
    std::vector<Type> predict_list;
    std::vector<Type> minus_weight;
    minus_weight.resize(weight_.size(), 0);
    for (size_t row = 0; row < data_set_.size();row++) {
      Type res = GetTotalWeight(data_set_[row].data);
      Type predict = Sigmoid(res);
      predict_list.emplace_back(predict);
      Type error = predict- data_set_[row].lable;
      for (size_t dt = 0; dt < data_set_[row].data.size(); dt++) {
        minus_weight[dt] += error * data_set_[row].data[dt];//(p-y)*xi
      }
    }
    for (size_t wi = 0; wi < weight_.size(); wi++) {
      weight_[wi] -= (1.0 / data_set_.size())*learning_rate_ * minus_weight[wi];
    }

    Type loss = GetLoss(predict_list);
    std::cout << "epoch:" << epoch << ",loss:" << loss << std::endl;
  }

  return true;
}

template<typename Type>
bool LogisticRegression<typename Type>::TrainSgd() {
  learning_rate_ = 1;
  for (int epoch = 0; epoch < epoch_; epoch++) {
    std::vector<Type> predict_list;
    for (size_t row = 0; row < data_set_.size(); row++) {
      Type res = GetTotalWeight(data_set_[row].data);
      Type predict = Sigmoid(res);
      predict_list.emplace_back(predict);
      Type gradient = predict - data_set_[row].lable;
      for (size_t wi = 0; wi < weight_.size(); wi++) {
        weight_[wi] -= (1.0/ data_set_.size())*learning_rate_ * gradient*data_set_[row].data[wi];
      }
    }
    Type loss = GetLoss(predict_list);
    std::cout << "epoch:" << epoch << ",loss:" << loss << std::endl;
  }

  return true;
}

template<typename Type>
Type LogisticRegression<typename Type>::GetTotalWeight(const std::vector<Type>& data){
  Type res = 0;
  if (weight_.size() != data.size()){
    return res;
  }
  for (size_t dt = 0; dt < data.size();dt++) {
    res += weight_[dt] * data[dt];
  }

  return res;
}

template<typename Type>
Type LogisticRegression<typename Type>::Predict(const std::vector<Type>& input_data){
  Type res = GetTotalWeight(input_data);
  Type predict = Sigmoid(res);
  if (predict >= 0.5){
    return 1.0;
  }
  else{
    return 0.0;
  }
}

}
