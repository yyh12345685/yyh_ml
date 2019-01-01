#include <iostream>
#include "ada_boost.h"

int main(){
  std::vector<std::vector<float>> train_data;
  std::vector<float>row = { 1,5 ,1};
  std::vector<float>row1 = { 2,2 ,1};
  std::vector<float>row2 = { 3,1 ,-1};
  std::vector<float>row3 = { 4,6 ,-1};
  std::vector<float>row4 = { 6,8 ,1};
  std::vector<float>row5 = { 6,5 ,-1};
  std::vector<float>row6 = { 7,9 ,1};
  std::vector<float>row7 = { 8,7,1};
  std::vector<float>row8 = { 9,8 ,-1};
  std::vector<float>row9 = { 10,2 ,-1};
  train_data.push_back(row);
  train_data.push_back(row1);
  train_data.push_back(row2);
  train_data.push_back(row3);
  train_data.push_back(row4);
  train_data.push_back(row5);
  train_data.push_back(row6);
  train_data.push_back(row7);
  train_data.push_back(row8);
  train_data.push_back(row9);

  std::vector<float>test_row = { 1,1 };
  std::vector<float>test_row1 = { 10,10 };
  std::vector<std::vector<float>> test_data;
  test_data.push_back(test_row);
  test_data.push_back(test_row1);

  ada_boost::AdaBoost ada_boost;
  ada_boost.Train(train_data, train_data.size(), train_data[0].size());

  std::vector<int>result;
  ada_boost.Classify(result, test_data);
  for (int idx = 0; idx < test_data.size();idx++) {
    std::cout << "test line:";
    for (const auto& it : test_data[idx]){
      std::cout << it << "\t";
    }
    std::cout << "classify:" << result[idx] << std::endl;
  }
  return 0;
}