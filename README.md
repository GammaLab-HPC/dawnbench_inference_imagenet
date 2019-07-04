## Performance

**latency without preprocess:**  0.82ms

**latency with preprocess:** 1.26ms

**top5 error:**  0.0699



## Enviroment Prepare

**Host:**  1 nvidia Tesla T4 ,  1 Xeon(G) Gold 6130 cpu ,  128G cpu memory

**Software:**  ubuntu 16.04 , nvidia driver 410.104 , cuda 10.0 , cudnn 7.5 , tensorrt 5.1.5.0

**Model:**  caffe model resnet50



## Usage

Modify 'model_path, imgs_path, gt_path'  in test.cpp.

```c++
mkdir build && cd build && cmake ..
make
./AIProject
```
