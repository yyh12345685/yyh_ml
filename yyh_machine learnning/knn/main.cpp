#include "knn.h"

int main(){
  std::cout << "ÇëÊäÈëk£º";
  int k;
  std::cin >> k;
  std::cout << std::endl;
  knn::Knn<float,std::string> knn("data.txt", k);
  knn.Train();
  knn.GetLable();

  return 0;
}
