#include <fstream>
#include <stdlib.h>
#include <iostream>
#include "random_forest.h"
#include "common.h"
#include "card_tree.h"

namespace random_forest {

  bool RandomForest::InitAndTrain(const std::string& train_path, const std::string& test_path){
    if (!ReadData(train_path,test_path)){
      return false;
    }

    InitData();

    Train();
    return true;
  }

  void RandomForest::Predict(const std::string& out_path){
    std::fstream fs; //�ļ�д����
    fs.open(out_path.c_str(), std::fstream::out | std::fstream::trunc);

    for (int idx = 0; idx < test_row_; ++idx) {
      double max = -9999;
      int max_index = -9999;
      for (size_t jdx = 0; jdx < types_.size(); ++jdx) {
        if (max < for_train_out[idx][jdx]) {
          max = for_train_out[idx][jdx];
          max_index = jdx;
        }
      }
      //������
      fs << idx << "," << max_index+1 << std::endl;
    }
  }

  void RandomForest::InitData() {
    train_in_row_ = train_in_data_.size();
    train_in_col_ = train_in_data_[0].size();
    std::vector<int> tmp;
    tmp.resize(types_.size(), 0);
    transform_out_.assign(train_in_row_, tmp);
    for (size_t idx = 0; idx < train_out_data_.size(); ++idx) {
      //����Ϊ1��26������Ϊ0��25���ڶ���ά�ȼ�ȥ1
      transform_out_[idx][train_out_data_[idx]-1] = 1; 
    }

    test_row_ = test_data_.size();
    test_col_ = test_data_[0].size();

    std::vector<double> dtmp;
    dtmp.resize(types_.size(), 0);
    for_train_out.resize(test_row_,dtmp);
  }

  bool RandomForest::ReadData(const std::string& train_path, const std::string& test_path){
    return ReadTrain(train_path) && ReadTest(test_path);
  }

  bool RandomForest::ReadTrain(const std::string& train_path){
    std::ifstream read_fs(train_path);
    if (!read_fs.is_open()) {
      return false;
    }
    //ȥ����һ��
    std::string line_string;
    std::getline(read_fs, line_string);

    while (std::getline(read_fs, line_string)) {
      std::vector<double> line_item;
      common::Split(line_item, line_string, ',', true);
      std::vector<double>::iterator last = --line_item.end();//flag
      train_out_data_.emplace_back(*(last));
      types_.insert(*(last));
      line_item.erase(line_item.begin());//ɾ����һ��id
      line_item.resize(line_item.size() - 1);//ɾ�����һ��
      train_in_data_.emplace_back(line_item);
    }
    read_fs.close();
    return true;
  }

  bool RandomForest::ReadTest(const std::string& test_path){
    std::ifstream test_read_fs(test_path);
    if (!test_read_fs.is_open()) {
      return false;
    }
    //ȥ����һ��
    std::string line_string_test;
    std::getline(test_read_fs, line_string_test);

    while (std::getline(test_read_fs, line_string_test)) {
      std::vector<double> line_item;
      common::Split(line_item, line_string_test, ',', true);
      line_item.erase(line_item.begin());//ɾ����һ��
      test_data_.emplace_back(line_item);
    }
    test_read_fs.close();
    return true;
  }

  int RandomForest::Train(){
    std::cout << "start train.........." << std::endl;
    srand((unsigned)time(NULL));
    for (int times = 0; times < FOREST_SIZE ; times++) {
      std::cout << "train times:"<<times << std::endl;
      std::vector<std::vector<double> > sampling_in_data;
      std::vector<std::vector<int> > sampling_transform_out;
      //һ������������,�зŻصĳ飨������޷Żصģ������������n�����֣�����֮��ѡ��ǰm����
      for (size_t sampling = 0; sampling < train_in_data_.size() / 2;sampling++) {
        int rand = std::rand() % train_in_data_.size();
        sampling_in_data.emplace_back(train_in_data_[rand]);
        sampling_transform_out.emplace_back(transform_out_[rand]);
      }
        
      TrainInner(sampling_in_data, sampling_transform_out);
    }

    return 0;
  }

  int RandomForest::TrainInner(
    std::vector<std::vector<double> >& sampling_in_data,
    std::vector<std::vector<int> >& sampling_transform_out){

    for (const auto type:types_) {
      if (type == types_.size() / 2) {//for debug
        std::cout << "handle half in one train" << std::endl;
      }
      //���������������������
      std::vector<int> one_type_out;
      for (size_t idx = 0; idx < sampling_in_data.size(); ++idx) {
        one_type_out.emplace_back(sampling_transform_out[idx][type-1]);
      }

      DecisionTree decision_tree;
      decision_tree.CreativeDecisionTree(sampling_in_data, one_type_out);

      for (int jdx = 0; jdx < test_col_; ++jdx) {
        for_train_out[jdx][type-1] += decision_tree.Predict(test_data_[jdx]);//Ԥ��
      }
    }

    return 0;
  }

}
