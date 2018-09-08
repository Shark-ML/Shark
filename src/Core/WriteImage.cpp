#define SHARK_COMPILE_DLL
#include <shark/Core/DLLSupport.h>
#include <shark/Core/Images/WriteImage.h>
#include <memory>
#include <fstream>
#include <boost/algorithm/string.hpp>
extern "C"{
#include <png.h>
#include <zlib.h>
#include <jpeglib.h>
}
using namespace shark;
	
		
////////////////////////PNG WRITE///////////////////////
template<class T>
std::vector<unsigned char> image::writePNG(blas::dense_vector_adaptor<T const> const& image, Shape const& shape, PixelType type){
	RANGE_CHECK(shape.size() == 3);
	SHARK_RUNTIME_CHECK(image.size() == shape.numElements(), "Shape does not match input vector size");
	
	//convert pixel type to png color type
	int colorType = 0;
	switch(type){
	case PixelType::RGB:
		colorType = PNG_COLOR_TYPE_RGB;
		SHARK_RUNTIME_CHECK(shape[0] == 3, "RGB image requires 3 input channels");
	break;
	case PixelType::RGBA:
	case PixelType::ARGB:
		colorType = PNG_COLOR_TYPE_RGB_ALPHA;
		SHARK_RUNTIME_CHECK(shape[0] == 4, "RGBA image requires 4 input channels");
	break;
	case PixelType::Luma:
		colorType = PNG_COLOR_TYPE_GRAY;
		SHARK_RUNTIME_CHECK(shape[0] == 1, "Luma image requires 1 input channel");
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
		png, info, (png_uint_32)shape[2], (png_uint_32)shape[1], 8, colorType,  
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT//last 2 parameters must be set to this
	);
	png_write_info(png, info);
	
	
	//encode the image one line at a time
	std::vector<png_byte> line(shape[0] * shape[2]);//width * channels
	
	//convert ARGB to RGBA
	std::size_t index_channels_RGBA[]={0,1,2,3};
	std::size_t index_channels_ARGB[]={3,0,1,2};
	std::size_t* index_c = index_channels_RGBA;
	if(type == PixelType::ARGB){
		index_c = index_channels_ARGB;
	}
	for(std::size_t i = 0; i != shape[1]; ++i){
		for(std::size_t c = 0; c != shape[0]; ++c){
			std::size_t imagePos =  c * shape[1] * shape[2] + i * shape[2];
			for(std::size_t j = index_c[c]; j < line.size(); j += shape[0], ++imagePos){
				line[j] = (unsigned char)(image[imagePos] * 255 + T(0.5)); 
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
std::vector<unsigned char> image::writeJPEG(blas::dense_vector_adaptor<T const> const& image, Shape const& shape, PixelType type){
	RANGE_CHECK(shape.size() == 3);
	SHARK_RUNTIME_CHECK(image.size() == shape.numElements(), "Shape does not match input vector size");
	
	//convert pixel type to png color type
	J_COLOR_SPACE colorType;
	switch(type){
	case PixelType::RGB:
		colorType = JCS_RGB;
		SHARK_RUNTIME_CHECK(shape[0] == 3, "RGB image requires 3 input channels");
	break;
	case PixelType::RGBA:
	case PixelType::ARGB:
		throw SHARKEXCEPTION("JPEG does not support alpha channels");
	break;
	case PixelType::Luma:
		colorType = JCS_GRAYSCALE;
		SHARK_RUNTIME_CHECK(shape[0] == 1, "Luma image requires 1 input channel");
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
		info.image_width = (JDIMENSION) shape[2];
		info.image_height = (JDIMENSION) shape[1];
		info.input_components = (JDIMENSION) shape[0];
		info.in_color_space = colorType;
		
		
		//setup compression configuration
		jpeg_set_defaults(&info);
		jpeg_set_quality(&info, 100, false);
		jpeg_start_compress(&info, TRUE);
		
		std::vector<unsigned char> line(shape[0] * shape[2]);
		while (info.next_scanline < info.image_height) {
			for(std::size_t c = 0; c != shape[0]; ++c){
				std::size_t imagePos =  c * shape[1] * shape[2] + info.next_scanline * shape[2];
				for(std::size_t j = c; j < line.size(); j += shape[0], ++imagePos){
					line[j] = (unsigned char)(image[imagePos] * 255 + T(0.5)); 
				}
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
std::vector<unsigned char> image::writePGM(blas::dense_vector_adaptor<T const> const& image, Shape const& shape, PixelType type){
	RANGE_CHECK(shape.size() == 3);
	SHARK_RUNTIME_CHECK(image.size() == shape.numElements(), "Shape does not match input vector size");
	SHARK_RUNTIME_CHECK(type == PixelType::Luma, "PGM does only support Luma Pixeltype");
	SHARK_RUNTIME_CHECK(shape[0] == 1, "PGM images cn only have one channel");
	
	//create header
	std::string header = "P5\n" + std::to_string(shape[2]) + ' ' + std::to_string(shape[1]) + "\n255\n";
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
void image::writeImageToFile(std::string const& filename, blas::dense_vector_adaptor<T const> const& image, Shape const& shape, PixelType type){
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



//explicit instantiation
template SHARK_EXPORT_SYMBOL std::vector<unsigned char> image::writePNG<float>(blas::dense_vector_adaptor<float const> const& image, Shape const& shape, PixelType type);
template SHARK_EXPORT_SYMBOL std::vector<unsigned char> image::writePNG<double>(blas::dense_vector_adaptor<double const> const& image, Shape const& shape, PixelType type);
template SHARK_EXPORT_SYMBOL std::vector<unsigned char> image::writeJPEG<float>(blas::dense_vector_adaptor<float const> const& image, Shape const& shape, PixelType type);
template SHARK_EXPORT_SYMBOL std::vector<unsigned char> image::writeJPEG<double>(blas::dense_vector_adaptor<double const> const& image, Shape const& shape, PixelType type);
template SHARK_EXPORT_SYMBOL std::vector<unsigned char> image::writePGM<float>(blas::dense_vector_adaptor<float const> const& image, Shape const& shape, PixelType type);
template SHARK_EXPORT_SYMBOL std::vector<unsigned char> image::writePGM<double>(blas::dense_vector_adaptor<double const> const& image, Shape const& shape, PixelType type);
template SHARK_EXPORT_SYMBOL void image::writeImageToFile(std::string const& filename, blas::dense_vector_adaptor<float const> const& image, Shape const& shape, PixelType type);
template SHARK_EXPORT_SYMBOL void image::writeImageToFile(std::string const& filename, blas::dense_vector_adaptor<double const> const& image, Shape const& shape, PixelType type);