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

#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

#include "cuproc.h"


#include <opencv2/highgui.hpp>
#include <iostream>

int main(int argc, const char **argv)
{

    // std::ifstream fin("fill_list.bin", std::ios::binary);

    // char fill_list[512*512];
    // fin.read(fill_list, 512*512);
    // for (int i = 0; i < 512; i++) {

    //     for (int j = 0; j < 512; j++) {
    //         std::cout << int(fill_list[i*512+j]) << " \t";

    //     }
    //     std::cout<< std::endl;
    // }
    // fin.close();

    // FILE* file_list = fopen("../fill_list.bin", "rb");
    // uchar fill_list[512*512];
    
	// if( !file_list ) {
    //     std::cout << "ERROR in file read\n";
	// 	return -1;
	// }

	// // read the font file
	// const size_t ttf_read = fread(fill_list, 1, 512*512, file_list);

    //  for (int i = 0; i < 512; i++) {
    //     for (int j = 0; j < 512; j++) {
    //         std::cout << int(fill_list[i*512+j]) << " \t";
    //     }
    //     std::cout<< std::endl;
    // }

	// fclose(file_list);


    Mat frame;
    // 000087.jpg
    std::string video {"/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/%06d.jpg"};//= argv[1];
    VideoCapture cap(video);

    // get bounding box
    cap >> frame;
    cv::resize(frame , frame, cv::Size2i(1920, 1080));
    CudaProcess cup(frame);
    auto ptr_ = cup.getImgPtr();

    auto font = cudaFont::CreateWithBorder();

    // font->OverlayText((uchar4 *) ptr_, 1920, 1080,"Hello world!", 100, 100, make_float4(255, 0, 0 ,255) );
    font->OverlayText((uchar4 *) ptr_, IMAGE_RGBA8, 1920, 1080, 
                                        "Hello World abcdefgh! nb", 100, 100, make_uchar4(255, 0, 0, 255), make_uchar4(0, 0, 255, 255), 0 );
    
    auto img = cup.backToImage();


    cv::imshow( "img ", img );
    cv::waitKey(0);

    return -1;




    static const uint32_t MaxCommands = 1024;
	static const uint32_t FirstGlyph  = 32; // 32
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


    std::ifstream gtIfstream("../GlyphInfo.txt");
    std::string gtLine;
    for (int n = 0; n < NumGlyphs; n++) {
        getline(gtIfstream, gtLine);
        std::stringstream gtStream(gtLine);
        std::string element;
        std::vector<int> elements;

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].x = uint16_t(std::atof(element.c_str()));

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].y = uint16_t(std::atof(element.c_str()));  
        
        std::getline(gtStream, element, ',');
        mGlyphInfo[n].width = uint16_t(std::atof(element.c_str()));  

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].height = uint16_t(std::atof(element.c_str()));  

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].xAdvance = float(std::atof(element.c_str()));  

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].xOffset = float(std::atof(element.c_str()));

        std::getline(gtStream, element, ',');
        mGlyphInfo[n].yOffset = float(std::atof(element.c_str()));

        // std::cout << gtLine << std::endl;
    }

    gtIfstream.close();

    std::cout << "n" << "\t" << "c" << "\t" 
        << "x" << "\t" << "y" << "\t"
        << "w" << "\t" << "h" << "\t"
        << "xA" << "\t\t" << "xO" << "\t"
        << "yO" << std::endl;

    for( uint32_t n=0; n < NumGlyphs; n++ ) {
        char temp_char = n + 32;
        std::cout << n + 32 << "\t" << temp_char << "\t" 
        << mGlyphInfo[n].x << "\t" << mGlyphInfo[n].y << "\t"
        << mGlyphInfo[n].width << "\t" << mGlyphInfo[n].height << "\t"
        << mGlyphInfo[n].xAdvance << "\t\t" << mGlyphInfo[n].xOffset << "\t"
        << mGlyphInfo[n].yOffset << std::endl;
    }




    return -1;
}


