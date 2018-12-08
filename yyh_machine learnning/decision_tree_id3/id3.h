#pragma once

#include <set>
#include "common.h"

namespace id3{

 const static std::string lable_yes = "yes";
 const static std::string lable_no = "no";

struct Node {//决策树节点
  std::string attribute;//属性值
  std::string arrived_value;//到达该属性之前的值
  std::vector<Node *> childs;//所有的孩子
};

class DecisionTreeID3 {
public:
  DecisionTreeID3():root_(nullptr) {
  }
  ~DecisionTreeID3() {
    if (root_!=nullptr){
      FreeNode(root_);
    }
  }
  //加载数据
  bool LoadData(const std::string& file_name);
  //创建决策树
  Node* CreativeTree();
  //结果预测
  std::string Predict(const std::vector<std::string>& item);
  //打印树
  void PrintTree();
protected:

  std::string PredictInner(Node* node, const std::vector<std::string>& item);

  Node* CreativeTreeInner(Node* node, 
    std::vector<std::vector<std::string> >& data, std::vector<std::string>& attributes);

  void PrintTreeInner(Node* node, int depth);

  bool IsLeaf(Node* node, std::vector<std::vector<std::string> >& data);

  //选择最大的信息增益列
  int ChooseMaxGainColumn(std::vector<std::vector<std::string> >& data);

  double ComputeEntropy(std::vector<std::vector<std::string> >& data);

  //获取列特征列表
  std::set<std::string> GetFeatureColumnList(
    std::vector< std::vector<std::string> > &data, int column);

  std::string SelectMoreLable(std::vector< std::vector<std::string> > &data);

  //按照给定特征划分数据集，划分后的数据集中不包含给定特征，即新的数据集删除了一列
  std::vector< std::vector<std::string> > GetFeatureRemoveColumn(
    const std::vector< std::vector<std::string> >& data, int column, const std::string& value);

  void FreeNode(Node* node) {
    if (node == nullptr){
      return;
    }
    for (size_t idx =0;idx< node->childs.size();idx++){
      FreeNode(node->childs[idx]);
    }
    delete node;
  }
private:
  Node* root_;
  std::vector<std::string> attributes_;//属性列
  std::vector<std::vector<std::string> > data_;
};

}
