
#include "random_forest.h"

int main(){
  random_forest::RandomForest random_forest;
  random_forest.InitAndTrain("train.csv", "test.csv");
  random_forest.Predict("test.result");
  return 0;
}

