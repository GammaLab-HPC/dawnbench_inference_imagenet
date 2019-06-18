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

class GammaAPI DetectContext
{
public:
  DetectContext(std::string path, int gpuid);
  ~DetectContext();
public:
  bool isStop() const;
  bool isEmpty() const;
  void setStop(bool flag);
  int gpuid() const;

  void addImage(std::string imagepath);
  void addImage2(std::string imagepath, int start, int end);
  bool fetchImage(std::vector<cv::Mat> & images);

  void addImage3(std::string imagepath, int start, int end);
  bool fetchImage2(std::unordered_map<int, cv::Mat> & images);

  std::string getimgpath();

private:
  int gpuid_;
  bool stopFlag_;
  std::string imgpath_;
private:

  std::vector<cv::Mat> images_;
  std::unordered_map<int, cv::Mat> imagesmap_;

private:
  std::mutex lockImages_;//...
  std::condition_variable cvImages_;//...
private:
  std::thread * thLoadImage_;
};



class GammaAPI ImageNet2
{
public:
  ImageNet2();
  ~ImageNet2();
  bool load_models(const std::string & modpath, int gpu = 0);
  bool inference(const std::vector<cv::Mat>& src, std::vector<std::vector<float>>& drc);
  bool inference_gpu(const std::vector<cv::Mat>& src, std::vector<std::vector<int>>& drc);
  bool set_device(int dev);

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

  bool pre_process(const std::vector<cv::Mat> & src, std::vector<cv::Mat> & dst);

};

