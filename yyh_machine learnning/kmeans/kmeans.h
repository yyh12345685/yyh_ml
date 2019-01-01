#pragma once

#include "common.h"
#include <fstream>

namespace kmeans{

  template<typename T>
  class Kmeans{
  public:
    Kmeans(int k);
    bool LoadData(const char *read_file_name);
    void KmeansKernel();
  private:
    std::vector< std::vector<T> > data_list_;//����
    int col_len_;//�еĳ���
    int row_len_;//����
    int k_kinds_;//���������
    std::vector< std::vector<T> > cent_points_;

    typedef struct MinMax{
      T min;//ÿ��ά�ȵ���Сֵ
      T max;//ÿ��ά�ȵ����ֵ
      MinMax(T min, T max) :min(min), max(max) {}
    }MinMax;

    typedef struct NodeTypeInfo{
      int min_index; //ÿһ���ڵ������������ĵ㣬�����ڵڼ���
      double min_dist;//����������ĵ�ľ���
      NodeTypeInfo(int idx, double dist) :min_index(idx), min_dist(dist) {}
    }NodeTypeInfo;

    std::vector<NodeTypeInfo>  data_type_info_;
    void RandCent();//��ʼ�������ѡ�����ĵ�
    void InitNodeTypeInfo();//����ÿ���ڵ�ĳ�ʼֵ
    MinMax GetMinMax(int idx);
    void SetCent(MinMax &tminmax, int idx);
    double Dist(std::vector<T> &v1, std::vector<T> &v2);
    void Debug();
  };

  template<typename T>
  Kmeans<T>::Kmeans(int k) {
    k_kinds_ = k;
  }

  template<typename T>
  void Kmeans<T>::InitNodeTypeInfo(){
    NodeTypeInfo tmp(-1,-1.0);
    data_type_info_.resize(row_len_, tmp);
    //memset(&(data_type_info_[0]), '\0', sizeof(T)*row_len_);
  }

  template<typename T>
  void Kmeans<T>::SetCent(MinMax &minmax, int idx){
    T range = minmax.max - minmax.min;
    for (int ix = 0; ix < k_kinds_; ix++){
      /* generate float data between 0 and 1 */
      cent_points_[ix].at(idx) = minmax.min + range * (rand() / (double)RAND_MAX);
    }
  }

  //��ȡ��idxά�ȵ����ֵ����Сֵ
  template<typename T>
  typename Kmeans<T>::MinMax Kmeans<T>::GetMinMax(int idx){
    T min = data_list_[0].at(idx);
    T max = min;

    for (int ii = 1; ii < row_len_; ii++){
      if (data_list_[ii].at(idx) < min)
        min = data_list_[ii].at(idx);
      else if (data_list_[ii].at(idx) > max)
        max = data_list_[ii].at(idx);
      else 
        continue;
    }

    MinMax minmax(min, max);
    return minmax;
  }

  //��ά�ռ���������ľ���
  template<typename T>
  double Kmeans<T>::Dist(std::vector<T> &v1, std::vector<T> &v2){
    T sum = 0;
    int size = v1.size();
    for (int i = 0; i < size; i++){
      sum += (v1[i] - v2[i])*(v1[i] - v2[i]);
    }
    return sum;
  }

  template<typename T>
  void Kmeans<T>::Debug(){
    std::cout << "new cent point:" << std::endl;
    auto itc = cent_points_.begin();
    while (itc != cent_points_.end()){
      auto it2c = (*itc).begin();
      while (it2c != (*itc).end()){
        std::cout << *it2c << "\t";
        it2c++;
      }
      std::cout << std::endl;
      itc++;
    }
    std::cout << std::endl;
    std::cout << "data and grouping:" << std::endl;
    typename std::vector< std::vector<T> > ::iterator it = data_list_.begin();
    typename std::vector< NodeTypeInfo > ::iterator itt = data_type_info_.begin();
    for (int row = 0; row < row_len_; row++){
      typename std::vector<T> ::iterator it2 = (*it).begin();
      while (it2 != (*it).end()){
        std::cout << *it2 << "\t";
        it2++;
      }
      std::cout << (*itt).min_index << std::endl;
      itt++;
      it++;
    }
    std::cout << std::endl;
  }

  template<typename T>
  void Kmeans<T>::RandCent(){
    std::vector<T> vec(col_len_, 0);
    for (int ik = 0; ik < k_kinds_; ik++){
      cent_points_.push_back(vec);
    }

    srand(time(NULL));
    for (int jj = 0; jj < col_len_; jj++){
      MinMax minmax = GetMinMax(jj);
      SetCent(minmax, jj);
    }
  }

  template<typename T>
  bool Kmeans<T>::LoadData(const char *read_file_name){
    std::ifstream read_file(read_file_name);
    if (!read_file.is_open()) {
      std::cout << "Open read file fail,file_path:" << read_file_name << std::endl;
      return false;
    }
    std::string line_string;
    while (std::getline(read_file, line_string)) {
      std::vector<T> line_item;
      common::Split(line_item, line_string, '\t', false);
      data_list_.push_back(line_item);
    }

    row_len_ = data_list_.size();
    col_len_ = data_list_[0].size();

    return true;
  }

  template<typename T>
  void Kmeans<T>::KmeansKernel(){
    //��ʼ��һЩ����
    RandCent();
    InitNodeTypeInfo();

    bool changed = true;//�ı����ĵ����������Ƿ�ı�
    int iter_times = 1;
    while (changed){
      changed = false;
      //����һ���ҳ����������ĵ�����ĵ㣬������
      std::cout << "find the nearest to cent of each point item:"<<iter_times<< std::endl;
      for (int row = 0; row < row_len_; row++){
        int min_index = -1;
        double min_dist = INT_MAX;
        for (int jdx = 0; jdx < k_kinds_; jdx++){
          double distji = Dist(cent_points_[jdx], data_list_[row]);
          if (distji < min_dist){
            min_dist = distji;
            min_index = jdx;//�ֵ���jdx��
          }
        }
        if (data_type_info_[row].min_index != min_index){//����ı���
          changed = true;
          data_type_info_[row].min_index = min_index;
          data_type_info_[row].min_dist = min_dist;
        }
      }

      //��������ҵ��µ�������ĵ�
      std::cout << "update the cent points:" << std::endl;
      for (int cent = 0; cent < k_kinds_; cent++){
        std::vector<T> vec(col_len_, 0);
        int cnt = 0;
        for (int row = 0; row < row_len_; row++){
          if (data_type_info_[row].min_index == cent){
            ++cnt;
            //sum of two vectors
            for (int col = 0; col < col_len_; col++){
              vec[col] += data_list_[row].at(col);
            }
          }
        }

        //���ĵ�Ϊ��һ��������У�ÿһ�������֮���ƽ��ֵ
        for (int ii = 0; ii < col_len_; ii++){
          if (cnt != 0)	
            vec[ii] /= cnt;
          cent_points_[cent].at(ii) = vec[ii];
        }
      }//for
      Debug();
      iter_times++;
    }//while
  }

}
