#pragma once

#include <string>
#include <vector>

//ÿ�ηָ�ʱѡȡ���еĸ�����������н����ģ���˵�����ʱ������ֵӦΪ sqrt��numparameters - 2��
//���ʱ����ֵΪ��numparametres - 2�� / 3��������Ȼ����ϣ�����Ϊ�˼��ټ��㣬����ѡ����ǰ�ߡ�
#define SELECTED_COLUMNS 40
#define MAX_DEEP_NUM 7//����������

namespace random_forest {

struct TreeNode{
  std::vector<int>data_index; //���ڸýڵ�����е��е�id
  //���ڼ�¼�ýڵ�ķָ��
  std::vector<double>attributes[SELECTED_COLUMNS];
  TreeNode* left_child = nullptr;
  TreeNode* right_child = nullptr;
  int is_leaf = 0;// 0 ��Ҷ�ӽڵ㣬1 ��ʾҶ�ӽڵ�
  int split_label;//��¼��ǰ�ķָ����Ե�λ�ã�������һ��
  double label_value;//�ýڵ�ķָ�ֵ                 
  double value;//�ýڵ��ƽ��ֵ���������ʱ������
  int deep;//���ڼ�¼���ڲ��������
  double gini;//��ǰ�ڵ��GINIֵ��������洢�ˣ���ʵûɶ��
};

class DecisionTree{
public:
  bool CreativeDecisionTree(
    const std::vector<std::vector<double> >& train_in_data, 
    const std::vector<int>&  train_out_data);

  double Predict(const std::vector<double>& line);

  DecisionTree(){
    root_ = nullptr;
  }
  ~DecisionTree(){
    ReleaseTree(root_);
  }

protected:

  struct MinginiInfo {
    int par_label = -1;
    double par_value = -1;
    double min_gini = 99999;
  };

  double PredictInner(TreeNode* node,const std::vector<double>& line);

  void ComputeChildDataIndex(
    MinginiInfo& min_gini_info,TreeNode* node, TreeNode* left_child, TreeNode* right_child);

  void GetMinGini(TreeNode* node, std::vector<int>& selected_col_vec,MinginiInfo& min_gini_info);
  //���ڼ���ָ�ڵ�ʱ����Ҫ��giniֵ
  double ComputeGini(TreeNode* node, int label, double value);
  double GetLeafValue(TreeNode* node);
  void ComputePerNodeGini(TreeNode* node);
  void SelectedClos(std::vector<int>& selected_col, int select_cnt);
  void GetNodeAttr(std::vector<int>& selected_col, TreeNode* node);

  void CreativeChildNode(TreeNode* node);
  bool CheckNodeIsLeaf(TreeNode* node);

protected:
  void ReleaseTree(TreeNode* node){
    if (node->left_child != nullptr){
      ReleaseTree(node->left_child);
    }
    if (node->right_child != nullptr){
      ReleaseTree(node->right_child);
    }
    delete node;
  }
private:
  std::vector<std::vector<double> > train_in_data_;
  std::vector<double>  train_lable_;

  TreeNode* root_;
};

}
