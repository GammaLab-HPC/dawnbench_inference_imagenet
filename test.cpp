#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>


#ifdef WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include "GammaContext.h"

std::string model_path = "/path/to/resnet50_model/";
std::string imgs_path = "/path/to/ILSVRC2012_val/";
std::string gt_path = "/path/to/gt.txt";

std::vector<int> load_gt(const std::string &path)
{
  std::ifstream file(path);
  std::vector<int> gt;
  std::string value;
  while (std::getline(file, value, '\n'))
  {
    gt.push_back(std::stoi(value));
  }
  return gt;
}

float eval(std::unordered_map<int, std::vector<int>> det)
{
  std::vector<int> gt = load_gt(gt_path);

  if (gt.size() != det.size())
  {
    std::cout << "gt size (" << gt.size() <<")!= det size(" << det.size() <<  ")" << std::endl;
    return -1;
  }

  const int nums = gt.size();
  int err = 0;
  for (int i=0; i<nums; i++)
  {
    int flag = 0;
    for (int j=0; j<det[i+1].size(); j++)
    {
      if (gt[i] == det[i+1][j])
      {
        flag = 1;
        break;
      }
    }
    if (!flag) err++;
  }

  float top5 = (float)err / (float)nums;
  return top5;
}

int main(int argc, char* argv[])
{
  if (argc == 4)
  {
  }

#ifndef WIN32
  struct timeval timestart;
  struct timeval timeend;
#endif

  std::unordered_map<int, std::vector<int>> final_res;
  final_res.reserve(50000);
  std::vector<std::vector<int>> res;

  /*imagenet resnet50 demo*/
  ImageNet2 imagenet;
  imagenet.load_models(model_path, 0);

#ifdef WIN32
  DWORD start = GetTickCount();
#else
  gettimeofday(&timestart, NULL);
#endif

  std::cout << "start" << std::endl;
  DetectContext DC(imgs_path, 0);

  while((!DC.isStop()) || (!DC.isEmpty())){
    std::unordered_map<int, cv::Mat> images;
    if(!DC.fetchImage2(images)) continue;
    for(std::unordered_map<int, cv::Mat>::iterator it = images.begin(); it!= images.end(); it++)
    {
      imagenet.inference_gpu({it->second},res);
      final_res[it->first] = res[0];
    }
  }

#ifdef WIN32
  std::cout << GetTickCount() - start << std::endl;
#else
  gettimeofday(&timeend, NULL);
  double m_dectime = (timeend.tv_sec - timestart.tv_sec) * 1000 + (timeend.tv_usec - timestart.tv_usec) / 1000;
  std::cout << "inference time : " << m_dectime << std::endl;
#endif

  float top5 = eval(final_res);
  std::cout << top5 << std::endl;

  return 0;
}
