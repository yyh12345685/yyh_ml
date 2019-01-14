#pragma once

#include <vector>
#include <set>

//���ﶨ��������ɭ�ֵ�size�� ��ע�⣬�����ɭ���У�ÿtypesNum�ö��������ľۺϲű���Ϊ�����ɭ���е�һ������
//��ˣ�������ɭ�ֵ�sizeΪ 1000�� ��ôʵ���ϻṹ�� 1000 * typesNum�ö�������
//#define FOREST_SIZE 1000 
#define FOREST_SIZE 6 //��ʽģ������Ҫ����Ϊ1000���ң�����Ϊ�˽�ʡʱ����ٳ����

namespace random_forest{

class RandomForest{
public:
  bool InitAndTrain(const std::string& train_path, const std::string& test_path);
  void Predict(const std::string& out_path);
protected:
  bool ReadData(const std::string& train_path, const std::string& test_path);
  bool ReadTrain(const std::string& train_path);
  bool ReadTest(const std::string& test_path);

  void InitData();
  int Train();
  int TrainInner(
    std::vector<std::vector<double> >& sampling_in_data, 
    std::vector<std::vector<int> >& sampling_transform_out);

private:
  std::vector<std::vector<double> > train_in_data_;
  std::vector<double> train_out_data_;

  std::set<int> types_;

  std::vector<std::vector<int> > transform_out_;
  int train_in_row_;
  int train_in_col_;

  std::vector<std::vector<double> > test_data_;
  int test_row_;
  int test_col_;

  std::vector<std::vector<double> >for_train_out;

};

}

