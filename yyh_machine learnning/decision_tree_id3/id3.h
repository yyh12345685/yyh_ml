#pragma once

#include <set>
#include "common.h"

namespace id3{

 const static std::string lable_yes = "yes";
 const static std::string lable_no = "no";

struct Node {//�������ڵ�
  std::string attribute;//����ֵ
  std::string arrived_value;//���������֮ǰ��ֵ
  std::vector<Node *> childs;//���еĺ���
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
  //��������
  bool LoadData(const std::string& file_name);
  //����������
  Node* CreativeTree();
  //���Ԥ��
  std::string Predict(const std::vector<std::string>& item);
  //��ӡ��
  void PrintTree();
protected:

  std::string PredictInner(Node* node, const std::vector<std::string>& item);

  Node* CreativeTreeInner(Node* node, 
    std::vector<std::vector<std::string> >& data, std::vector<std::string>& attributes);

  void PrintTreeInner(Node* node, int depth);

  bool IsLeaf(Node* node, std::vector<std::vector<std::string> >& data);

  //ѡ��������Ϣ������
  int ChooseMaxGainColumn(std::vector<std::vector<std::string> >& data);

  double ComputeEntropy(std::vector<std::vector<std::string> >& data);

  //��ȡ�������б�
  std::set<std::string> GetFeatureColumnList(
    std::vector< std::vector<std::string> > &data, int column);

  std::string SelectMoreLable(std::vector< std::vector<std::string> > &data);

  //���ո��������������ݼ������ֺ�����ݼ��в������������������µ����ݼ�ɾ����һ��
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
  std::vector<std::string> attributes_;//������
  std::vector<std::vector<std::string> > data_;
};

}
