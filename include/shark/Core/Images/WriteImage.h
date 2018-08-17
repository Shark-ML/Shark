//===========================================================================
/*!
 * 
 *
 * \brief       Exporting of Images
 * 
 * 
 *
 * \author      O.Krause
 * \date        2018
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#ifndef SHARK_DATA_WRITE_IMAGE_H
#define SHARK_DATA_WRITE_IMAGE_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
#include <shark/Core/Images/Enums.h>
#include <memory>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
extern "C"{
#include <png.h>
#include <zlib.h>
#include <jpeglib.h>
}


namespace shark { namespace image{
	
	
////////////////////////PNG WRITE///////////////////////
template<class T>
std::vector<unsigned char> writePNG(blas::vector<T> const& image, Shape const& shape, PixelType type){
	RANGE_CHECK(shape.size() == 3);
	SHARK_RUNTIME_CHECK(image.size() == shape.numElements(), "Shape does not match input vector size");
	
	//convert pixel type to png color type
	int colorType = 0;
	switch(type){
	case PixelType::RGB:
		colorType = PNG_COLOR_TYPE_RGB;
		SHARK_RUNTIME_CHECK(shape[2] == 3, "RGB image requires 3 input channels");
	break;
	case PixelType::RGBA:
	case PixelType::ARGB:
		colorType = PNG_COLOR_TYPE_RGB_ALPHA;
		SHARK_RUNTIME_CHECK(shape[2] == 4, "RGBA image requires 4 input channels");
	break;
	case PixelType::Luma:
		colorType = PNG_COLOR_TYPE_GRAY;
		SHARK_RUNTIME_CHECK(shape[2] == 1, "Luma image requires 1 input channel");
	}
	
	//preallocate buffer
	std::vector<unsigned char> buffer;
	buffer.reserve(image.size()+100);//allocate enough space so that we do not need to reallocate
	
	//initialize png structures
	png_struct*  png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	SHARK_RUNTIME_CHECK(png, "Out of Memory!");
	png_info* info = png_create_info_struct(png);
	if( !info ){
		png_destroy_write_struct(&png, NULL);
		throw SHARKEXCEPTION("Out of Memory!");
	}
	//libpngs hacky way to do error reporting...
	if (setjmp(png_jmpbuf(png))) {
		png_destroy_read_struct(&png, &info, nullptr);
		throw SHARKEXCEPTION("An error occured while processing the PNG file");
	}	
	
	//setup io
	auto write =[](png_struct* png, png_byte* data, png_size_t length){//append bytes to buffer
		auto buffer = (std::vector<unsigned char>*)png_get_io_ptr(png);
		buffer->insert(buffer->end(), data, data + length);//can not fail due to earlier reserve
	};
	auto flush = [](png_struct*){};//nothing to do
	png_set_write_fn(png, &buffer, write, flush);
	
	//activate best compression
	png_set_compression_level(png, Z_BEST_COMPRESSION);

	//write image header
	png_set_IHDR(
		png, info, shape[1], shape[0], 8, colorType,  
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT//last 2 parameters must be set to this
	);
	png_write_info(png, info);
	
	
	//encode the image one line at a time
	std::vector<png_byte> line(shape[1]*shape[2]);
	std::size_t imagePos = 0;
	for(std::size_t i = 0; i != shape[0]; ++i){
		//convert ARGB to RGBA
		if(type == PixelType::ARGB){
			for(std::size_t j = 0; j != line.size(); j+=4, imagePos +=4){
				line[j] = png_byte(image[imagePos + 1] * 255); 
				line[j + 1] = (unsigned char)(image[imagePos + 2] * 255 + T(0.5)); 
				line[j + 2] = (unsigned char)(image[imagePos + 3] * 255 + T(0.5)); 
				line[j + 3] = (unsigned char)(image[imagePos] * 255 + T(0.5)); 
			}
		}else{
			for(std::size_t j = 0; j != line.size(); ++j, ++imagePos){
				line[j] = (unsigned char)(image[imagePos]*255 +T(0.5));
			}
		}
		png_write_row(png, line.data());
	}
	png_write_end(png, NULL);
	png_destroy_write_struct(&png, &info);
	
	return buffer;
}


/////////////////////////////////////JPEG WRITE/////////////////////////////////////////////////////

template<class T>
std::vector<unsigned char> writeJPEG(blas::vector<T> const& image, Shape const& shape, PixelType type){
	RANGE_CHECK(shape.size() == 3);
	SHARK_RUNTIME_CHECK(image.size() == shape.numElements(), "Shape does not match input vector size");
	
	//convert pixel type to png color type
	J_COLOR_SPACE colorType;
	switch(type){
	case PixelType::RGB:
		colorType = JCS_RGB;
		SHARK_RUNTIME_CHECK(shape[2] == 3, "RGB image requires 3 input channels");
	break;
	case PixelType::RGBA:
	case PixelType::ARGB:
		throw SHARKEXCEPTION("JPEG does not support alpha channels");
	break;
	case PixelType::Luma:
		colorType = JCS_GRAYSCALE;
		SHARK_RUNTIME_CHECK(shape[2] == 1, "Luma image requires 1 input channel");
	}
	
	
	//initialize jpeg compression objects
	struct jpeg_compress_struct info;
	jpeg_error_mgr mgr;
	info.err = jpeg_std_error(&mgr);
	mgr.error_exit = [](j_common_ptr info){throw info->err;};
	jpeg_create_compress(&info);
	
	//setup memory io
	unsigned char* buffer;
	unsigned long outsize = 0;
	jpeg_mem_dest(&info, &buffer, &outsize);
	
	try{
		//setup file header
		info.image_width = shape[1];
		info.image_height = shape[0];
		info.input_components = shape[2];
		info.in_color_space = colorType;
		
		
		//setup compression configuration
		jpeg_set_defaults(&info);
		jpeg_set_quality(&info, 100, false);
		jpeg_start_compress(&info, TRUE);
		
		std::vector<unsigned char> line(shape[1]*shape[2]);
		while (info.next_scanline < info.image_height) {
			auto row = subrange(image,info.next_scanline * line.size(), (info.next_scanline + 1) * info.next_scanline); 
			for(std::size_t j = 0; j != line.size(); ++j){
				line[j] = (unsigned char)(row[j]*255+T(0.5));
			}
			unsigned char* lines[1] = {line.data()};
			jpeg_write_scanlines(&info, lines, 1);
		}
		
		jpeg_finish_compress(&info);
	}catch(jpeg_error_mgr* mgr){//on failure, write message and rethrow
		char messageBuffer[JMSG_LENGTH_MAX];
		( *(mgr->format_message) ) ((j_common_ptr)&info, messageBuffer);
		jpeg_abort_compress(&info);
		jpeg_destroy_compress(&info);
		free(buffer);
		throw SHARKEXCEPTION(std::string("[readJPEG] Error during reading:") + messageBuffer);
	}
	jpeg_destroy_compress(&info);
	
	std::vector<unsigned char> ret(buffer, buffer+outsize);
	free(buffer);
	
	return ret;
}

///////////////////////////////////PGM WRITE///////////////////////////////////////////////////////

template<class T>
std::vector<unsigned char> writePGM(blas::vector<T> const& image, Shape const& shape, PixelType type){
	RANGE_CHECK(shape.size() == 3);
	SHARK_RUNTIME_CHECK(image.size() == shape.numElements(), "Shape does not match input vector size");
	SHARK_RUNTIME_CHECK(type == PixelType::Luma, "PGM does only support Luma Pixeltype");
	SHARK_RUNTIME_CHECK(shape[2] == 1, "PGM images cn only have one channel");
	
	//create header
	std::string header = "P5\n" + std::to_string(shape[1]) + ' ' + std::to_string(shape[0]) + "\n255\n";
	//preallocate buffer
	std::vector<unsigned char> buffer;
	buffer.reserve(image.size()+header.size());
	//copy header as unsigned chars
	for(char c: header)
		buffer.push_back((unsigned char)c);
	
	for(T value: image){
		unsigned char pix = (unsigned char)(value * 255+T(0.5));
		buffer.push_back(pix);
	}
	return buffer;
}

template<class T>
void writeImageToFile(std::string const& filename, blas::vector<T> const& image, Shape const& shape, PixelType type){
	//find extension
	std::size_t pos = filename.find_last_of('.');
	SHARK_RUNTIME_CHECK(pos != std::string::npos, "Can not find extension");
	auto extension = boost::algorithm::to_lower_copy(filename.substr(pos + 1));
	
	//convert image by extension
	std::vector<unsigned char> buffer;
	if(extension == "png")
		buffer = writePNG(image, shape, type);
	else if(extension == "pgm")
		buffer = writePGM(image, shape, type);
	else if(extension == "jpeg" || extension == "jpg")
		buffer = writeJPEG(image, shape, type);
	else
		throw SHARKEXCEPTION("Unknown image format");
	
	std::ofstream file(filename, std::ios::binary);
	SHARK_RUNTIME_CHECK(file, "Could not open file!");
	
	file.write((char*) buffer.data(), buffer.size());
}



}}
#endif
