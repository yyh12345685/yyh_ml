#pragma once

#include <string>
#include <vector>

//每次分割时选取的列的个数，这个是有讲究的，据说分类的时候它的值应为 sqrt（numparameters - 2）
//拟合时它的值为（numparametres - 2） / 3，我们虽然是拟合，但是为了加速计算，还是选用了前者。
#define SELECTED_COLUMNS 40
#define MAX_DEEP_NUM 7//树的最大深度

namespace random_forest {

struct TreeNode{
  std::vector<int>data_index; //属于该节点的所有的行的id
  //用于记录该节点的分割点
  std::vector<double>attributes[SELECTED_COLUMNS];
  TreeNode* left_child = nullptr;
  TreeNode* right_child = nullptr;
  int is_leaf = 0;// 0 非叶子节点，1 表示叶子节点
  int split_label;//记录当前的分割属性的位置，属于哪一列
  double label_value;//该节点的分割值                 
  double value;//该节点的平均值，用于拟合时的运算
  int deep;//用于记录所在层数，深度
  double gini;//当前节点的GINI值，算出来存储了，其实没啥用
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
  //用于计算分割节点时所需要的gini值
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
