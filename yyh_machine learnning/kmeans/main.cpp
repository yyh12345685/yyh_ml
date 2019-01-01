#include "kmeans.h"
#include <iostream>

int main(int argc, char *argv[]){

  if (argc != 3){
    std::cout << "Usage : ./a.out filename k" << std::endl;
    return -1;
  }

  const char *filename = argv[1];
  int kind = atoi(argv[2]);

  kmeans::Kmeans<double> kmeans(kind);
  if (!kmeans.LoadData(filename)){
    return -2;
  }

  kmeans.KmeansKernel();
  return 0;
}
