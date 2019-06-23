#pragma once

#define GammaAPI

#include <mutex>
#include <condition_variable>
#include <list>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <thread>
#include <set>
#include <opencv2/opencv.hpp>

class GammaAPI ImageNet2
{
public:
  ImageNet2();
  ~ImageNet2();
  std::unordered_map<int, cv::Mat> readPictureVec3(std::string path, int start, int end);
  bool load_models(const std::string & modpath, int gpu = 0);
  bool inference(const std::vector<cv::Mat>& src, std::vector<std::vector<float>>& drc);
  bool inference_gpu(const std::vector<cv::Mat>& src, std::vector<std::vector<int>>& drc);
  bool inference_gpu(float* src, std::vector<int>& drc);
  bool set_device(int dev);
  bool pre_process(cv::Mat & src, float* dst);
  bool pre_process(cv::Mat & src, float* & dst);

private:
  void * trt_core_;
  int gpuid_;

  float* mean_gpu_;
  float* std_gpu_;
  float* netoutput_gpu_;
  float* inputdata_gpu_;

  float* multip_output_;
  float* subtract_output_;
  float* crop_output_;
  float* resize_output_;
  unsigned char* inputmat_;
  unsigned char* inputmat2_;

  float* data_gpu_;

  bool pre_process(const std::vector<cv::Mat> & src, std::vector<cv::Mat> & dst);

};
