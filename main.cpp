#include <iostream>
#include "cudaFont.h"

#include <string>


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cudaFont.h"
#include "cudaVector.h"
#include "cudaOverlay.h"
#include "cudaMappedMemory.h"

// #include "imageIO.h"
#include "filesystem.h"
#include "logging.h"

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"
#include "cudaAlphaBlend.cuh"
#include <vector>

using namespace std;

int main()
{
    static const uint32_t MaxCommands = 1024;
	static const uint32_t FirstGlyph  = 32;
	static const uint32_t LastGlyph   = 255;
	static const uint32_t NumGlyphs   = LastGlyph - FirstGlyph;

    struct GlyphInfo
	{
		uint16_t x;
		uint16_t y;
		uint16_t width;
		uint16_t height;

		float xAdvance;
		float xOffset;
		float yOffset;
	} mGlyphInfo[NumGlyphs];

    GlyphInfo mGlyphInfo_BOLD[NumGlyphs];
    int mFontMapWidth;
	int mFontMapHeight;
    mFontMapWidth  = 512;
	mFontMapHeight = 512;

    uint8_t* mFontMapCPU;
	uint8_t* mFontMapGPU;

    uint8_t* mFontMapCPU_BOLD;
	uint8_t* mFontMapGPU_BOLD;


    float size = 40.0f;


    string filename = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf";
    const size_t ttf_size = fileSize(filename);

    void* ttf_buffer = malloc(ttf_size);
    FILE* ttf_file = fopen(filename.c_str(), "rb");

    const size_t ttf_read = fread(ttf_buffer, 1, ttf_size, ttf_file);

	fclose(ttf_file);
    stbtt_bakedchar bakeCoords[NumGlyphs];

    const size_t fontMapSize = mFontMapWidth * mFontMapHeight * sizeof(unsigned char);

    cudaAllocMapped((void**)&mFontMapCPU, (void**)&mFontMapGPU, fontMapSize);

    const int result = stbtt_BakeFontBitmap((uint8_t*)ttf_buffer, 0, size, 
										mFontMapCPU, mFontMapWidth, mFontMapHeight,
									    FirstGlyph, NumGlyphs, bakeCoords);

    cv::Mat im(mFontMapHeight, mFontMapWidth, CV_8UC1,mFontMapCPU);
    cv::Mat img_bgr;
    cv::cvtColor( im, img_bgr, cv::COLOR_GRAY2BGR); 

    cv::imshow( "img_bgr", img_bgr );

    cv::waitKey(0);

    // string filename_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf";
    // const size_t ttf_size_BOLD = fileSize(filename_BOLD);

    // void* ttf_buffer_BOLD = malloc(ttf_size_BOLD);
    // FILE* ttf_file_BOLD = fopen(filename_BOLD.c_str(), "rb");

    // const size_t ttf_read_BOLD = fread(ttf_buffer_BOLD, 1, ttf_size_BOLD, ttf_file_BOLD);

	// fclose(ttf_file_BOLD);
    // stbtt_bakedchar bakeCoords_BOLD[NumGlyphs];

    // // const size_t fontMapSize = mFontMapWidth * mFontMapHeight * sizeof(unsigned char);

    // cudaAllocMapped((void**)&mFontMapCPU_BOLD, (void**)&mFontMapGPU_BOLD, fontMapSize);

    // const int result_BOLD = stbtt_BakeFontBitmap((uint8_t*)ttf_buffer_BOLD, 0, size, 
	// 									mFontMapCPU_BOLD, mFontMapWidth, mFontMapHeight,
	// 								    FirstGlyph, NumGlyphs, bakeCoords_BOLD);

    // cout<<result<<endl;

    // for( uint32_t n=0; n < NumGlyphs; n++ )
	// {
	// 	mGlyphInfo[n].x = bakeCoords[n].x0;
	// 	mGlyphInfo[n].y = bakeCoords[n].y0;

	// 	mGlyphInfo[n].width  = bakeCoords[n].x1 - bakeCoords[n].x0;
	// 	mGlyphInfo[n].height = bakeCoords[n].y1 - bakeCoords[n].y0;

	// 	mGlyphInfo[n].xAdvance = bakeCoords[n].xadvance;
	// 	mGlyphInfo[n].xOffset  = bakeCoords[n].xoff;
	// 	mGlyphInfo[n].yOffset  = bakeCoords[n].yoff;

    //     mGlyphInfo_BOLD[n].x = bakeCoords_BOLD[n].x0;
	// 	mGlyphInfo_BOLD[n].y = bakeCoords_BOLD[n].y0;

	// 	mGlyphInfo_BOLD[n].width  = bakeCoords_BOLD[n].x1 - bakeCoords_BOLD[n].x0;
	// 	mGlyphInfo_BOLD[n].height = bakeCoords_BOLD[n].y1 - bakeCoords_BOLD[n].y0;

	// 	mGlyphInfo_BOLD[n].xAdvance = bakeCoords_BOLD[n].xadvance;
	// 	mGlyphInfo_BOLD[n].xOffset  = bakeCoords_BOLD[n].xoff;
	// 	mGlyphInfo_BOLD[n].yOffset  = bakeCoords_BOLD[n].yoff;

    //     cout<<n<<" "
    //     <<mGlyphInfo_BOLD[n].width <<" "
    //     <<mGlyphInfo[n].width <<" "
    //     <<endl;

    //     // cout<<n<<" "
    //     //     <<mGlyphInfo[n].x<<" \t"
    //     //     <<mGlyphInfo[n].y<<" \t"
    //     //     <<mGlyphInfo[n].width<<" \t"
    //     //     <<mGlyphInfo[n].height<<" \t"
    //     //     <<mGlyphInfo[n].xOffset<<" \t"
    //     //     <<mGlyphInfo[n].yOffset<<" \t"<<endl;
	// }

    //  cudaMemcpy(frameBytes,frame_gpu, 1920*1080*4, cudaMemcpyDeviceToHost);

    
    // for( uint32_t n=0; n < NumGlyphs; n++ ) {
    //     if(mGlyphInfo[n].width > 0 && mGlyphInfo[n].height>0 ) {
    //         uint8_t char_pixels[mGlyphInfo[n].height][mGlyphInfo[n].width];

    //         // for(int)
    //     }
    // }


    // for( uint32_t n=0; n < NumGlyphs; n++ ){
    //     if(mGlyphInfo[n].width > 0 && mGlyphInfo[n].height>0 ) {
    //         cv::Mat img_bgr_crop;
    //         im(cv::Rect(mGlyphInfo[n].x ,mGlyphInfo[n].y, mGlyphInfo[n].width, mGlyphInfo[n].height)).copyTo(img_bgr_crop);
    //         cv::Mat img_bgr_crop_big, img_bgr_crop_pad;
    //         cv::resize(img_bgr_crop, img_bgr_crop_big, cv::Size(mGlyphInfo[n].width+8, mGlyphInfo[n].height+8));
            
    //         cv::copyMakeBorder(img_bgr_crop,img_bgr_crop_pad,4,4,4,4,cv::BORDER_ISOLATED,0);

    //         cout<<img_bgr_crop_pad.size()<<", "<<img_bgr_crop_big.size()<<endl;

    //         unsigned char new_img[img_bgr_crop_pad.rows*img_bgr_crop_pad.cols];

    //         for(int i{0}; i<img_bgr_crop_pad.rows; i++){
    //             for(int j{0}; j<img_bgr_crop_pad.rows; j++){
    //                 // cout<<int(img_bgr_crop_pad.at<unsigned char>(i,j))<<" ,";
    //                 // if(img_bgr_crop_pad.at<unsigned char>(i,j) ==255 && img_bgr_crop_big.at<unsigned char>(i,j)==0)  
    //                     // new_img[i*img_bgr_crop_pad.cols+j] = 255;
    //                      new_img[j*img_bgr_crop_pad.rows+j] = img_bgr_crop_pad.at<unsigned char>(i,j)*img_bgr_crop_big.at<unsigned char>(i,j)/256;      
    //             }
    //             // cout<<endl;
    //         }

    //         cv::Mat new_imgg(img_bgr_crop_pad.rows, img_bgr_crop_pad.cols, CV_8UC1,new_img);

    //         cv::Mat img_bgrr;
    //         cv::cvtColor( new_imgg, img_bgrr, cv::COLOR_GRAY2BGR); 
    //         cv::resize(img_bgrr,img_bgrr,cv::Size(mGlyphInfo[n].width*10, mGlyphInfo[n].height*10));

    //         cv::imshow( "img_bgrr", img_bgrr );

    //         // vector<cv::Mat> chs;
    //         // chs.push_back(img_bgr_crop_pad);
    //         // chs.push_back(img_bgr_crop_pad);
    //         // chs.push_back(img_bgr_crop_big);
    //         // cv::Mat merged_im ;
    //         // cv::merge(chs,merged_im);




    //         // cv::resize(merged_im,merged_im,cv::Size(mGlyphInfo[n].width*10, mGlyphInfo[n].height*10));
            
    //         // cv::imshow( "merged_im", merged_im );

    //         cv::waitKey(400);
    //     }
    // }








    // cv::Mat im(mFontMapHeight, mFontMapWidth, CV_8UC1,mFontMapCPU);
    // cv::Mat img_bgr;
    // cv::cvtColor( im, img_bgr, cv::COLOR_GRAY2BGR); 
    // cv::imshow( "frame", img_bgr );


    // cv::Mat im_BOLD(mFontMapHeight, mFontMapWidth, CV_8UC1,mFontMapCPU_BOLD);
    // cv::Mat img_bgr_BOLD;
    // cv::cvtColor( im_BOLD, img_bgr_BOLD, cv::COLOR_GRAY2BGR); 
    // cv::imshow( "frame_BOLD", img_bgr_BOLD );

    // vector<cv::Mat> chs;
    // chs.push_back(im);
    // chs.push_back(im_BOLD);
    // chs.push_back(im);
    // cv::Mat merged_im ;
    // cv::merge(chs,merged_im);
    
    // cv::imshow( "merged_im", merged_im );

    cv::waitKey(0);





    // cudaFont* m_font = cudaFont::Create(
    //     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    //     // "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf",
    //     40.0f); 

    // // cudaFont* m_font_bold = cudaFont::Create(
    // //     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    // //     // "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf",
    // //     42.0f); 

    // unsigned char *frameBytes;
    // frameBytes = (unsigned char *) malloc(1920*1080*4);
    // unsigned char *frame_gpu ;
    // cudaMalloc(&frame_gpu,1920*1080*4);
    // cudaMemcpy(frame_gpu, frameBytes, 1920*1080*4, cudaMemcpyHostToDevice);

    // stringstream stream;
    // stream<<"ABCDEFGHabcde"<<3.14<<"\xB0";
    // string str = stream.str();
    // // string str = "ABCDEFGHabcde"+"\xB0";//"fgh1234567890";    
    
    // // bool ret1 = m_font_bold->OverlayText(frame_gpu, IMAGE_RGBA8, 
    // //         1920, 1080, str.c_str(), 100, 100, make_float4(255,255,255, 255), 0 );
            

    // // ret1 = m_font_bold->OverlayText(frame_gpu, IMAGE_RGBA8, 
    // //         1920, 1080, str.c_str(), 200, 200, make_float4(255,255,255, 255), 0 );

            
    // // int4 bb_bold = m_font_bold->TextExtents(str.c_str());
    // // cout<<bb_bold.x<<" "<<bb_bold.y<<" "<<bb_bold.z<<" "<<bb_bold.w<<endl;



    // bool ret2 = m_font->OverlayText(frame_gpu, IMAGE_RGBA8, 
    //         300, 300,str.c_str() , 10+1, 10+1, make_float4(255,0,12, 255), 0 );


    // int4 bb_reg = m_font->TextExtents(str.c_str());
    // cout<<bb_reg.x<<" "<<bb_reg.y<<" "<<bb_reg.z<<" "<<bb_reg.w<<endl;


    // cudaMemcpy(frameBytes,frame_gpu, 300*300*4, cudaMemcpyDeviceToHost);
    // cv::Mat im(300, 300, CV_8UC4,frameBytes);
    // cv::Mat img_bgr;
    // cv::cvtColor( im, img_bgr, cv::COLOR_RGBA2BGR); 
    // cv::resize(img_bgr,img_bgr,cv::Size(1920,1080));
    // cv::imshow( "frame", img_bgr );
    // cv::waitKey(0);

    return -1;
}


