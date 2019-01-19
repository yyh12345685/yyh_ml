#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include "common.h"

namespace knn{

//Type1 数据的类型，Type2 标签的类型
template <typename Type1, typename Type2>
struct DataNode {
  std::vector<Type1> data;
  Type2 lable;
};

template <typename Type1, typename Type2>
class Knn{
public:
  Knn(const std::string& file_name, int k);
  void Train();
  Type2 GetLable();
protected:
  typedef std::pair<int, Type1> pair_val;
  struct CmpDistance
  {
    bool operator() (const pair_val& lhs, const pair_val& rhs)
    {
      return lhs.second < rhs.second;
    }
  };
  std::vector<pair_val>sort_distance_;

private:
  int k_;
  std::vector<DataNode<Type1, Type2> > data_set_;
  DataNode<Type1, Type2>  test_node_;//测试场景，用一个测试样本来测试
  bool LoadData(const std::string& file_name);
  Type1 GetDistance(const std::vector<Type1>& data1, const std::vector<Type1>& data2);
};

template<typename Type1, typename Type2>
Knn<typename Type1, typename Type2>::Knn(const std::string& file_name,int k) {
  k_ = k;
  if (!LoadData(file_name)){
    std::cout << "load data failed,file name:" << file_name << std::endl;
  }
  std::cout << "输入测试数据：" << std::endl;
  test_node_.data.resize(data_set_[0].data.size());
  for (size_t idx = 0; idx < data_set_[0].data.size(); idx++)
    std::cin >> test_node_.data[idx];
}

template<typename Type1, typename Type2>
bool Knn<typename Type1, typename Type2>::LoadData(const std::string& file_name){
  std::ifstream read_file(file_name);
  if (!read_file.is_open()) {
    std::cout << "Open read file fail,file_path:" << file_name << std::endl;
    return false;
  }
  std::string line_string;
  while (std::getline(read_file, line_string)) {
    std::vector<std::string> line_item;
    common::Split(line_item, line_string, ' ', false);
    if (line_item.size() <=1){
      continue;
    }
    DataNode<Type1, Type2> row;
    for (size_t idx = 0; idx < line_item.size() - 1;idx++) {
      row.data.emplace_back(atof(line_item[idx].c_str()));
    }
    
    row.lable = line_item[line_item.size()-1];
    data_set_.emplace_back(row);
  }

  if (data_set_.empty() || data_set_[0].data.empty()){
    return false;
  }
  return true;
}

template<typename Type1, typename Type2>
void Knn<typename Type1, typename Type2>::Train(){
  pair_val val;
  for (size_t ix = 0; ix < data_set_.size();ix++) {
    val.first = ix;
    val.second = GetDistance(data_set_[ix].data, test_node_.data);
    sort_distance_.emplace_back(val);
    std::cout << "index = " << val.first << " distance = " << val.second << std::endl;
  }
  std::sort(sort_distance_.begin(), sort_distance_.end(), CmpDistance());

}

template<typename Type1, typename Type2>
Type2 Knn<typename Type1, typename Type2>::GetLable(){
  std::map<Type2, int> lable_cnt;
  if (k_> (int)sort_distance_.size()){
    k_ = sort_distance_.size();
  }
  for (int jdx = 0; jdx < k_;jdx++) {
    std::cout << "the index = " << sort_distance_[jdx].first << " the distance = " 
      << sort_distance_[jdx].second << " the label = " << data_set_[sort_distance_[jdx].first].lable 
      << " the coordinate ( " << data_set_[sort_distance_[jdx].first].data[0] << "," 
      << data_set_[sort_distance_[jdx].first].data[1] << " )" << std::endl;
    lable_cnt[data_set_[sort_distance_[jdx].first].lable] ++;
  }

  int max_cnt = 0;
  Type2 test_lable;
  for (const auto lc : lable_cnt){
    if (lc.second > max_cnt){
      max_cnt = lc.second;
      test_lable = lc.first;
    }
  }

  std::cout << "test data belongs to the " << test_lable << " lable" << std::endl;
  return test_lable;
}

template<typename Type1, typename Type2>
Type1 Knn<typename Type1, typename Type2>::GetDistance(
  const std::vector<Type1>& data1, const std::vector<Type1>& data2){
  Type1 ret_val = 0.0;
  if (data1.size() != data2.size()){
    std::cout << "error data." << std::endl;
    return ret_val;
  }
  for (size_t idx = 0; idx < data1.size();idx++) {
    ret_val += pow(data1[idx] - data2[idx], 2);
  }
  return sqrt(ret_val);
}

}
