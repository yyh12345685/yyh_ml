
#include <map>
#include "id3.h"
#include <iostream>
#include <fstream>

namespace id3 {

  //加载数据
  bool DecisionTreeID3::LoadData(const std::string& file_name){
    std::ifstream read_file(file_name);
    if (!read_file.is_open()) {
      std::cout<<"Open read file fail,file_path:" << file_name;
      return false;
    }

    std::string line_string;
    bool first_line = true;
    while (std::getline(read_file, line_string)) {
      std::vector<std::string> line_item;
      common::Split(line_item, line_string, ' ', true);
      if (line_item.size() < 6)
        continue;
      if (first_line){
        //属性不包括结果列
        for (size_t idx = 1; idx < line_item.size()-1; idx++) {
          attributes_.push_back(line_item[idx]);
          std::cout << line_item[idx] << "\t";
        }
        std::cout << std::endl;
        first_line = false;
      }
      else {
        std::vector<std::string>line;
        for (size_t jdx = 1; jdx < line_item.size(); jdx++) {
          line.push_back(line_item[jdx]);
        }
        data_.push_back(line);
      }
    }
    return true;
  }
  //创建决策树
  bool DecisionTreeID3::CreativeTree() {
    if (attributes_.empty() || data_.empty()) {
      return false;
    }
    //check label
    std::vector<std::vector<std::string> >::iterator it = data_.begin();
    while (it!=data_.end()){
      int lab_idx = it->size() - 1;
      if ((*it)[lab_idx] != lable_yes && (*it)[lab_idx] !=lable_no){
        it = data_.erase(it);
      }
      else {
        it++;
      }
    } 
    root_ = new Node;
    CreativeTreeInner(root_,data_,attributes_);
    return true;
  }

  Node* DecisionTreeID3::CreativeTreeInner(Node* node,
    std::vector<std::vector<std::string> >& data, std::vector<std::string>& attributes){
    if (IsLeaf(node,data)){
      return node;
    }
    
    if (data.size() > 0 && 1 == data[0].size()){
      node->attribute = SelectMoreLable(data);
      return node;
    }
    //选择信息增益最大的列
    int idx = ChooseMaxGainColumn(data);
    std::string best_feature = attributes[idx];
    node->attribute = best_feature;
    //除去最大信息增益之后的属性特征
    std::vector<std::string> remaining_attributes;
    for (size_t jdx  = 0; jdx < attributes.size(); jdx++){
      if (best_feature != attributes[jdx])
        remaining_attributes.push_back(attributes[jdx]);
    }

    std::set<std::string> feature_list = GetFeatureColumnList(data, idx);
    for (const auto& item : feature_list) {
      Node *new_node = new Node();
      new_node->arrived_value = item;   //记录到达属性之前的取值
      //选择特征对应的行，并且是去掉了选择过的信息增益列，进行递归调用，生产子节点
      std::vector<std::vector<std::string> > sub_data = GetFeatureRemoveColumn(data, idx, item);
      CreativeTreeInner(new_node, sub_data, remaining_attributes);
      node->childs.push_back(new_node);
    }
    return node;
  }

  bool DecisionTreeID3::IsLeaf(Node* node, std::vector<std::vector<std::string> >& data) {
    int label_idx = data[0].size() - 1;
    std::set<std::string> labels;
    for (size_t idx = 0; idx < data.size();idx++) {
      labels.insert(data[idx][label_idx]);
    }
    if (labels.size() == 1){
      node->attribute = *(labels.begin());
      return true;
    }
    return false;
  }

  std::string DecisionTreeID3::SelectMoreLable(std::vector< std::vector<std::string> > &data){
    std::map<std::string, int> classify_counts;
    int label = data[0].size() - 1;
    for (size_t idx = 0; idx < data.size(); idx++) {
      if (classify_counts.find(data[idx][label]) != classify_counts.end()) {
        classify_counts[data[idx][label]]++;
      } else {
        classify_counts[data[idx][label]] = 0;
      }
    }
    int max_times = 0;
    std::string max_lable;
    for (const auto& it: classify_counts){
      if (it.second>max_times){
        max_times = it.second;
        max_lable = it.first;
      }
    }

    return max_lable;
  }

  int DecisionTreeID3::ChooseMaxGainColumn(std::vector<std::vector<std::string> >& data) {
    double entropy = ComputeEntropy(data);
    double max_gain = 0.0;
    double best_featrue_index = 0;
    //按照列遍历,不包括最后一列标签列
    for (size_t idx1 = 0; idx1 < data[0].size()-1;idx1++) {
      double sub_entropys = 0;
      std::set<std::string> feature_list = GetFeatureColumnList(data, idx1);
      for (const auto& item:feature_list){
        std::vector< std::vector<std::string> >sub_data = GetFeatureRemoveColumn(data, idx1, item);
        double prob = (double)sub_data.size() / (double)data.size();
        sub_entropys += prob * ComputeEntropy(sub_data);
      }
      double gain = entropy - sub_entropys;
      if (gain>max_gain){
        max_gain = gain;
        best_featrue_index = idx1;
      }
    }
    return best_featrue_index;
  }

  double DecisionTreeID3::ComputeEntropy(std::vector<std::vector<std::string> >& data) {
    std::map<std::string, int> classify_counts;
    int label = data[0].size() - 1;
    for (size_t idx = 0; idx < data.size(); idx++){
      if (classify_counts.find(data[idx][label])!= classify_counts.end()){
        classify_counts[data[idx][label]]++;
      }else{
        classify_counts[data[idx][label]] = 0;
      }
    }

    //计算香农熵
    double entropy = 0;
    for (const auto& it: classify_counts)
    {
      double prob = (double)(it.second) / (double)data.size();
      entropy -= prob * (log(prob) / log(2));
    }
    return entropy;
  }

  //获取列特征列表
  std::set<std::string> DecisionTreeID3::GetFeatureColumnList(
    std::vector< std::vector<std::string> > &data, int column)
  {
    std::set <std::string>feature_list;  //特征的所有取值
    for (size_t idx = 0; idx < data.size(); idx++)    //寻找该特征的所有可能取值
      feature_list.insert(data[idx][column]);
    return feature_list;
  }

//选取给定特征的行，按照给定特征划分数据集，
//划分后的数据集中不包含给定特征，即新的数据集删除了一列
 std::vector< std::vector<std::string> > DecisionTreeID3::GetFeatureRemoveColumn(
   const std::vector< std::vector<std::string> >& data, int column, const std::string& value)
{
   std::vector< std::vector<std::string> > result;
  for (size_t idx = 0; idx < data.size(); idx++)
  {
    if (data[idx][column] == value)
    {
      //将“当前特征”这个维度去掉
      std::vector<std::string> new_row(data[idx].begin(), data[idx].begin() + column);
      new_row.insert(new_row.end(), data[idx].begin() + column + 1, data[idx].end());
      result.push_back(new_row);
    }
  }
return result;
}

  //结果预测
  std::string DecisionTreeID3::Predict(const std::vector<std::string>& item) {
    return PredictInner(root_,item);
  }

  std::string DecisionTreeID3::PredictInner(Node* node, const std::vector<std::string>& item) {
    if (node->childs.size()==0){//叶子节点
      return node->attribute;
    }
    std::string attr = node->attribute;
    int node_feature_idx = 0;
    for (size_t idx = 0; idx <= attributes_.size();idx++) {
      if (attr == attributes_[idx]){
        node_feature_idx = idx;
        break;
      }
    }

    for (size_t chl_idx = 0; chl_idx < node->childs.size();chl_idx++) {
      if (item[node_feature_idx] == node->childs[chl_idx]->arrived_value){
        return PredictInner(node->childs[chl_idx], item);
      }
    }

    return "error Predict item";
  }

  void DecisionTreeID3::PrintTree(){
    int tab_time = 0;
    PrintTreeInner(root_, tab_time);
  }

  void DecisionTreeID3::PrintTreeInner(Node* node, int tab_time){
    for (int idx = 0; idx < tab_time;idx++) {
      std::cout << "\t";
    }
    if (node && !node->arrived_value.empty()){
      //如果有，打印到达之前的属性
      std::cout << node->arrived_value << std::endl;
      for (int idx = 0; idx < tab_time +1; idx++) {
        std::cout << "\t";
      }
    }
    std::cout << node->attribute << std::endl;
    for (const auto nd :node->childs){
      //打印子节点信息
      PrintTreeInner(nd, tab_time +2);
    }
  }
}

