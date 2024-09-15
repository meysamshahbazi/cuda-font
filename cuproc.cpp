#include "cuproc.h"

CudaProcess::CudaProcess(cv::Mat im) :im{im}
{

}

CudaProcess::~CudaProcess()
{

}

void* CudaProcess::getImgPtr()
{
    cv::Mat im_bgra;
    cv::cvtColor(im, im_bgra, cv::COLOR_BGR2RGBA);

    im_w = im_bgra.cols;
    im_h = im_bgra.rows;

    im_bgra_ptr = new uchar4[im_bgra.cols * im_bgra.rows];

    for(int i = 0; i < im_bgra.rows; i++) {
        for(int j = 0; j < im_bgra.cols; j++) {
            cv::Vec4b bgraPixel = im_bgra.at<cv::Vec4b>(i, j);
            uchar b_= bgraPixel[0];
            uchar g_= bgraPixel[1];
            uchar r_= bgraPixel[2];
            uchar a_= bgraPixel[3];
            uchar4 px = make_uchar4(b_, g_, r_, a_);
            im_bgra_ptr[i*im_bgra.cols + j] = px;
        }
    }

    cudaMalloc(&im_ptr, im_bgra.cols * im_bgra.rows* 4);
    cudaMemcpy(im_ptr, im_bgra_ptr, im_bgra.cols * im_bgra.rows* 4, cudaMemcpyHostToDevice);
    return im_ptr;
}

cv::Mat CudaProcess::backToImage() {
    cv::Mat img_bgr;
    cudaMemcpy(im_bgra_ptr, im_ptr, im_w*im_h*4, cudaMemcpyDeviceToHost);
    cv::Mat img = cv::Mat(im_h, im_w, CV_8UC4, im_bgra_ptr);
    cv::cvtColor(img, img_bgr, cv::COLOR_RGBA2BGR);
    return img_bgr;
}

void CudaProcess::freeImage()
{
    delete[] im_bgra_ptr;
    cudaFree(im_ptr);
}



