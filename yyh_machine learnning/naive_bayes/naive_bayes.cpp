#include <iostream>
#include <numeric>
#include "naive_bayes.h"

namespace naive_bayes {

  bool NaiveBayes::InitDoc(
    const std::vector<std::vector<std::string> >&doc_words,
    const std::vector<int>& classes){
    if (doc_words.size() != classes.size()) {
      return false;
    }
    doc_.doc_words = doc_words;
    doc_.classes = classes;
    int index = 1;
    for (const auto& doc :doc_.doc_words){
      for (const auto& word: doc){
        if (doc_.word_of_index.find(word)== doc_.word_of_index.end()){
          doc_.word_of_index[word] = index++;
        }
      }
    }
    return InitDocMatrix();
  }

  bool NaiveBayes::InitDocMatrix(){
    for (const auto& doc: doc_.doc_words){
      std::vector<int> vec(doc_.word_of_index.size() + 1,0);
      for (const auto& word : doc) {
        const auto& pit = doc_.word_of_index.find(word);
        if (pit != doc_.word_of_index.end() && pit->second < (int)(vec.size())) {
          vec[pit->second] = 1;
        }
      }
      doc_.doc_matrix.push_back(vec);
    }
    //debug
    std::cout << "print the train matrix begin : " << std::endl;
    for (const auto& vec : doc_.doc_matrix) {
      for (const auto& pt : vec) {
        std::cout << pt << " ";
      }
      std::cout << std::endl;
    }
    return true;
  }

  void NaiveBayes::Trainning(){
    if (0 == doc_.doc_matrix.size() || 0 == doc_.doc_matrix[0].size()){
      std::cout << "error,not init doc." << std::endl;
      return;
    }
    int docs = doc_.doc_matrix.size();
    std::cout << "num train docs:" << docs << std::endl;
    int sum = accumulate(doc_.classes.begin(), doc_.classes.end(), 0);
    p_abusive_ = (float)sum / (float)docs;
    std::cout << "sum=" << sum << ",p_abusive:" << p_abusive_ << std::endl;
    p0_vect_.resize(doc_.doc_matrix[0].size(), 1);
    p1_vect_.resize(doc_.doc_matrix[0].size(), 1);
    printf("p0_vect_.size() = %d , p1_vect_.size() = %d\n", p0_vect_.size(), p1_vect_.size());
    float p0_denom = 2.0; //the total number of words in non-abusive docs
    float p1_denom = 2.0; //the total number of words in abusive docs

    for (size_t idx = 0; idx < doc_.classes.size();idx++) {
      if (1 == doc_.classes[idx]){
        for (size_t jdx = 0; jdx < p1_vect_.size();jdx++) {
          //分类为1时，每个doc每个单词出现的次数的和
          p1_vect_[jdx] += doc_.doc_matrix[idx][jdx];
          if (1== doc_.doc_matrix[idx][jdx]){
            p1_denom++;
          }
        }
      }else{
        for (size_t jdx = 0; jdx < p0_vect_.size(); jdx++) {
          //分类为1时，每个doc每个单词出现的次数的和
          p0_vect_[jdx] += doc_.doc_matrix[idx][jdx];
          if (1 == doc_.doc_matrix[idx][jdx]) {
            p0_denom++;
          }
        }
      }
    }

    for (size_t jdx = 0; jdx < p1_vect_.size(); jdx++) {
      //分类为1时，每个doc每个单词出现的次数的和
      p1_vect_[jdx] = log(p1_vect_[jdx]/ p1_denom);
      p0_vect_[jdx] = log(p0_vect_[jdx] / p0_denom);
    }

    std::cout << "print the p1_vect_ values : " << std::endl;
    for (auto vec: p1_vect_)
      std::cout << vec << " ";
    std::cout << "\nprint the p0_vect_ values : " << std::endl;
    for (auto vec : p0_vect_)
      std::cout << vec << " ";
    std::cout << std::endl;
  }

  int NaiveBayes::Classify(const std::vector<std::string>& doc){
    std::vector<int> vec(doc_.word_of_index.size() + 1, 0);
    for (const auto& word : doc) {
      const auto& pit = doc_.word_of_index.find(word);
      if (pit != doc_.word_of_index.end() && pit->second < (int)(vec.size())) {
        vec[pit->second] = 1;
      }
    }
    for (auto& vc :vec)
      std::cout << vc << " ";
    std::cout << std::endl;
    //index start with one,so p1_vect_.begin() + 1
    float p1 = inner_product(p1_vect_.begin() + 1, p1_vect_.end(), vec.begin() + 1, float(0.0)) 
      + log(p_abusive_);
    float p0 = inner_product(p0_vect_.begin() + 1, p0_vect_.end(), vec.begin() + 1, float(0.0)) 
      + log(1 - p_abusive_);

    std::cout << "p1 = " << p1 << std::endl;
    std::cout << "p0 = " << p0 << std::endl;

    if (p1 > p0){
      return 1;
    } else{
      return 0;
    }
  }

}
