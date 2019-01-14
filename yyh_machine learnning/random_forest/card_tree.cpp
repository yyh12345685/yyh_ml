#include <set>
#include <algorithm>
#include <iostream>
#include "card_tree.h"

namespace random_forest {

  bool DecisionTree::CreativeDecisionTree(
    const std::vector<std::vector<double> >& train_in_data,
    const std::vector<int>&  train_lable){
    train_in_data_ = train_in_data;
    train_lable_.assign(train_lable.begin(), train_lable.end());
    root_ = new TreeNode;
    root_->deep = 1;
    for (size_t lb = 0; lb < train_lable.size(); lb++) {
      root_->data_index.emplace_back(lb);
    }
    //ComputePerNodeGini(root_);
    CreativeChildNode(root_);
    return true;
  }

  void DecisionTree::CreativeChildNode(TreeNode* node){
    if (CheckNodeIsLeaf(node)){
      return;
    }
    std::vector<int> selected_col;
    SelectedClos(selected_col, SELECTED_COLUMNS);
    GetNodeAttr(selected_col, node);
    
    MinginiInfo min_gini_info;
    GetMinGini(node, selected_col, min_gini_info);
    node->is_leaf = 0;
    node->split_label = min_gini_info.par_label;
    node->label_value = min_gini_info.par_value;

    node->left_child = new TreeNode();
    node->left_child->deep = node->deep+1;
    node->right_child = new TreeNode();
    node->right_child->deep = node->deep+1;
    ComputeChildDataIndex(min_gini_info, node, node->left_child, node->right_child);
    //ComputePerNodeGini(node->left_child);
    //ComputePerNodeGini(node->right_child);

    CreativeChildNode(node->right_child);
    CreativeChildNode(node->left_child);
  }
  
  void DecisionTree::ComputeChildDataIndex(
    MinginiInfo& min_gini_info,TreeNode* node, TreeNode* left_child, TreeNode* right_child){
    for (size_t idx = 0; idx < node->data_index.size(); ++idx) {
      if (train_in_data_[node->data_index[idx]][min_gini_info.par_label] <= min_gini_info.par_value) {
        left_child->data_index.emplace_back(node->data_index[idx]);
      } else {
        right_child->data_index.emplace_back(node->data_index[idx]);
      }
    }
  }

  void DecisionTree::GetMinGini(
    TreeNode* node, std::vector<int>& selected_col_vec, MinginiInfo& min_gini_info) {
    double temp_gini;
    for (size_t col = 0; col < selected_col_vec.size(); ++col) {
      std::vector<double>::iterator cursor = node->attributes[col].begin();
      for (; cursor != node->attributes[col].end(); ++cursor) {
        temp_gini = ComputeGini(node, selected_col_vec[col], *cursor);
        if (min_gini_info.min_gini > temp_gini) {
          min_gini_info.min_gini = temp_gini;
          min_gini_info.par_label = selected_col_vec[col];
          min_gini_info.par_value = *cursor;
        }
      }
    }
  }

  double DecisionTree::ComputeGini(TreeNode* node, int label, double value) {
    //һ����part��С�ڵ���value��һpart������value��һpart,category���ڼ�¼��part��Ԫ������
    int category1 = 0;
    int category2 = 0;//trainIn ����value�ĸ���
    int type10 = 0;//С�ڵ���valueʱ����Ϊ0�ĸ���
    int type11 = 0;//С�ڵ���valueʱ����Ϊ1�ĸ���
    int type20 = 0;//����valueʱ����Ϊ0�ĸ���
    int type21 = 0;//����valueʱ����Ϊ1�ĸ���
    int zeros = 0;//����Ϊ0�ĸ���

    unsigned int size = node->data_index.size();
    for (unsigned int ix = 0; ix < size; ++ix) {
      if (train_in_data_[node->data_index[ix]][label] <= value) {
        type10 += (train_lable_[node->data_index[ix]] == 0 ? 1 : 0);
        ++category1;
      }

      zeros += (train_lable_[node->data_index[ix]] == 0 ? 1 : 0);//����Ϊ0��
    }
    category2 = node->data_index.size() - category1;
    type20 = zeros - type10;
    type11 = category1 - type10;
    type21 = category2 - type20;

    double gini1 = 0;  //�������ֱַ��giniϵ��
    double gini2 = 0;

    if (category1 != 0)
      gini1 = 1 - pow((double)type10 / category1, 2) - pow((double)type11 / category1, 2);
    if (category2 != 0)
      gini2 = 1 - pow((double)type20 / category2, 2) - pow((double)type21 / category2, 2);
    return (double)category1 / size * gini1 + (double)category2 / size * gini2;
  }

  bool DecisionTree::CheckNodeIsLeaf(TreeNode* node){
    if ((node->gini > -0.000001 && node->gini < 0.000001) ||
      node->deep >= MAX_DEEP_NUM) {
      node->is_leaf = 1;
      node->value = GetLeafValue(node);
      return true;
    }
    return false;
  }

  void DecisionTree::SelectedClos(std::vector<int>& selected_col, int select_cnt){
    std::set<int> selected_col_set;
    while (selected_col_set.size() < (size_t)select_cnt){
      int rand = std::rand();
      selected_col_set.insert(rand%train_in_data_[0].size());
    }
    for (const auto& sel: selected_col_set){
      selected_col.emplace_back(sel);
    }
  }

  void DecisionTree::GetNodeAttr(std::vector<int>& selected_col, TreeNode* node){
    int sample_num = 100;
    //node->attributes��һά�ȱ�ʾ�������ڶ�ά�ȱ�ʾ����
    if (node->data_index.size() <= 100) {//С��100�Σ�ֱ�Ӱ��ն�����
      for (size_t idx = 0; idx < selected_col.size(); ++idx) {
        node->attributes[idx].resize(node->data_index.size());
        for (size_t jdx = 0; jdx < node->data_index.size(); jdx++) {
          node->attributes[idx][jdx] = train_in_data_[node->data_index[jdx]][selected_col[idx]];
        }
      }
      return;
    }

    //������forѭ����ѡ���40���������㣬ÿ�����Ե�ֵ��Ϊ100�Σ��洢��attributes��
    for (size_t idx = 0; idx < selected_col.size(); ++idx) {
      //ÿ������ѡȡ�ķָ��ĸ���������Ϊ100��
      node->attributes[idx].resize(100);
      std::vector<double> temp(node->data_index.size());//����ĳһ�е�����ֵ
      for (size_t jdx = 0; jdx < node->data_index.size(); ++jdx) {
        temp[jdx] = train_in_data_[node->data_index[jdx]][selected_col[idx]];
      }
      std::sort(temp.begin(), temp.end());//ѡ���ĳһ�е�ֵ��������
      //��������
      double factor = node->data_index.size() / 100;
      for (size_t kdx = 0; kdx < node->attributes[idx].size(); kdx++) {//��һ�е�����ֵ��Ϊ100��
        size_t cur_index = factor * kdx - int(factor * kdx) < 0.5 ? size_t(factor * kdx) : size_t(factor * kdx + 1);
        cur_index = cur_index > node->data_index.size() - 1 ? node->data_index.size() - 1 : cur_index;
        node->attributes[idx][kdx] = temp[cur_index];
      }
    }
  }

  double DecisionTree::GetLeafValue(TreeNode* node){
    double sum = 0;
    for (size_t iz = 0; iz < node->data_index.size(); ++iz) {
      sum += train_lable_[node->data_index[iz]]; //
    }
    return sum / node->data_index.size();//ƽ��ֵ
  }

  void DecisionTree::ComputePerNodeGini(TreeNode* node){
    int category1 = 0;  //��ǩΪ0������
    int category2 = 0; //��ǩΪ1������
    //���飬��1�ͷ�1��Ϊ����
    for (size_t idx = 0; idx < node->data_index.size(); ++idx) {
      if (train_lable_[node->data_index[idx]] == 0) {
        category1 += 1;
      }
    }
    category2 = node->data_index.size() - category1;
    node->gini = 1 - pow((double)category1 / node->data_index.size(), 2) 
      - pow((double)category2 / node->data_index.size(), 2);
  }

  double DecisionTree::Predict(const std::vector<double>& line){
    return PredictInner(root_, line);
  }

  double DecisionTree::PredictInner(TreeNode* node, const std::vector<double>& line){
    int static predict_times = 0;
    predict_times++;
    if (1 != node->is_leaf) {
      int col = node->split_label;
      double value = node->label_value;
      if (line[col] <= value) {
        return PredictInner(node->left_child, line);
      }
      else{
        return PredictInner(node->right_child, line);
      }
    }
    else{
      predict_times = 0;
      return node->value;
    }
    std::cout << "error,not should to here..." << std::endl;
  }

}
