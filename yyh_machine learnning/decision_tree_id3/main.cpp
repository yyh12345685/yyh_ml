#include <iostream>
#include "common.h"
#include "id3.h"

int main() {
  id3::DecisionTreeID3 id3;
  std::string  file("input.txt");
  if (!id3.LoadData(file)){
    std::cout << "LoadData failed,file:" << file << std::endl;
    return -1;
  }

  if (nullptr == id3.CreativeTree()){
    std::cout << "CreativeTree failed." << std::endl;
    return -2;
  }

  id3.PrintTree();

  std::vector<std::string> test = {"sunny", "hot",  "high", "strong"};

  std::cout << "²âÊÔ£º";
  for (size_t idx = 0; idx < test.size(); idx++) {
    std::cout << test[idx] << "\t";
  }
  std::cout << std::endl << "Ô¤²â£º";
  std::cout << id3.Predict(test);

  return 0;
}