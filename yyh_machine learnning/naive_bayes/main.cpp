#include <iostream>
#include "naive_bayes.h"

std::vector<std::vector<std::string> >train_doc_words = {
  {"my","dog","has","flea","problems","help","please"},
  {"maybe","not","take","him","to","dog","park","stupid"},
  {"my","dalmation","is","so","cute","I","love","him",},
  {"stop","posting","stupid","worthless","garbage"},
  {"mr","licks","ate","my","steak","how","to","stop","him"},
  {"quit","buying","worthless","dog","food","stupid"}
};

std::vector<int> train_classes = { 0, 1, 0, 1, 0, 1 };

std::vector<std::string> test_1 = { "love", "my", "dalmation" };
std::vector<std::string> test_2 = { "stupid", "garbage" };

int main() {
  //std::vector<std::string> test_21 = { "stupid", "garbage" };

  naive_bayes::NaiveBayes naive_bayes;
  naive_bayes.InitDoc(train_doc_words, train_classes);
  naive_bayes.Trainning();
  std::cout << "doc1 classified as : " << naive_bayes.Classify(test_1) << std::endl;
  std::cout << "doc2 classified as : " << naive_bayes.Classify(test_2) << std::endl;

  return 0;
}
