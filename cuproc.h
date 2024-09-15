
#ifndef _CUPROC_H_
#define _CUPROC_H_


#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>


using namespace std;

class CudaProcess
{
public:
    CudaProcess() = delete;
    CudaProcess(cv::Mat im);
    ~CudaProcess();
    void* getImgPtr();
    void freeImage();

    cv::Mat backToImage();

private:
    cv::Mat im;
    void* im_ptr;
    uchar4 *im_bgra_ptr;
    int im_w;
    int im_h;
};

#endif