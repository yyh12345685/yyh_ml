#include "logistic_regression.h"

int main(){
  //lr::LogisticRegression<float> lre(500);
  lr::LogisticRegression<float> lre(200);
  lre.InitData("logistic.txt");
  //lre.TrainBgd();
  lre.TrainSgd();

  std::vector<float> test;
  test.push_back((float)1);//½Ø¾à
  test.push_back((float)1.176813);
  test.push_back((float)3.167020);
  float res = lre.Predict(test);
  return 0;
}
