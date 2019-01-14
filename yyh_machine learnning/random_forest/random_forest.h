#pragma once

#include <vector>
#include <set>

//这里定义的是随机森林的size， 请注意，在随机森林中，每typesNum棵二分类树的聚合才被认为是随机森林中的一棵树。
//因此，如果随机森林的size为 1000， 那么实际上会构造 1000 * typesNum棵二分类树
//#define FOREST_SIZE 1000 
#define FOREST_SIZE 6 //正式模型中需要设置为1000左右，这里为了节省时间快速出结果

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

