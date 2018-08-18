#define SHARK_COMPILE_DLL
#include <shark/Core/DLLSupport.h>
#include <shark/Core/Images/ReadImage.h>
#include <memory>
#include <cstdlib>
#include <cctype>
#include <fstream>
extern "C"{
#include <png.h>
#include <jpeglib.h>
}


using namespace shark;
	
	
////////////////////////PNG READER///////////////////////
template<class T>
std::pair<blas::vector<T>, Shape> image::readPNG(std::vector<unsigned char> const& data){
	//Let LibPNG check the sig. If this function returns 0, everything is OK.
	if(png_sig_cmp((unsigned char*)data.data(), 0, 8) != 0)
		throw SHARKEXCEPTION("[readPNG] Image does not look like a png");
	
	png_struct* pngPtr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	if (!pngPtr) {
		throw SHARKEXCEPTION("[readPNG] Couldn't initialize png read struct");
	}
	png_info* infoPtr = png_create_info_struct(pngPtr);
	if (!infoPtr) {
		png_destroy_read_struct(&pngPtr, nullptr, nullptr);
		throw SHARKEXCEPTION("[readPNG] Couldn't initialize png info struct");
	}
	
	//libpngs hacky way to do error reporting...
	if (setjmp(png_jmpbuf(pngPtr))) {
		png_destroy_read_struct(&pngPtr, &infoPtr, nullptr);
		throw SHARKEXCEPTION("[readPNG] An error occured while processing the PNG file");
	}

	//set up the data source callback
	struct pngIOSource{
		unsigned char const* data;
		std::size_t pos;
	};
	auto readPNGData = [](png_struct* pngPtr, unsigned char* data, long unsigned int length) {
		pngIOSource* source = (pngIOSource*)png_get_io_ptr(pngPtr);
		for(std::size_t i = 0; i != length; ++i, ++source->pos){
			data[i] = source->data[source->pos];
		}
	};
	pngIOSource source ={data.data(), 0};
	png_set_read_fn(pngPtr,(void*)&source, readPNGData);
	
	//read the image
	png_read_png(
		pngPtr, infoPtr, 
		PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND, 
		NULL
	);
	
	//get image geometry
	std::size_t imgWidth =  png_get_image_width(pngPtr, infoPtr);
	std::size_t imgHeight = png_get_image_height(pngPtr, infoPtr);
	std::size_t channels = png_get_channels(pngPtr, infoPtr);
	
	//pixel contents
	auto rows = png_get_rows(pngPtr, infoPtr);
	
	//convert to float vector
	T conv = (T)255.0;
	blas::vector<T> image(imgWidth * imgHeight * channels);
	for (std::size_t i=0; i != imgHeight; ++i){
		for (std::size_t j=0; j != imgWidth * channels; ++j){
			image[i * imgWidth * channels +j] = rows[i][j]/conv;
		}
	}
	
	//done
	return {std::move(image), {imgHeight, imgWidth, channels}};
	
}

////////////////////////JPEG READER///////////////////////

template<class T>
std::pair<blas::vector<T>, Shape> image::readJPEG(std::vector<unsigned char> const& data){
	//initialize jpeg object and configure error reporting via exceptions
	jpeg_decompress_struct cinfo;
	jpeg_error_mgr mgr;
	cinfo.err = jpeg_std_error(&mgr);
	mgr.error_exit = [](j_common_ptr cinfo){throw cinfo->err;};
	
	//try to decompress (this can call our error callback defined above)
	try{	
		//set up decompression
		jpeg_create_decompress(&cinfo);
		
		//set up the data source
		jpeg_mem_src(&cinfo, (unsigned char*)data.data(), data.size());
		
		//check if the file is a valid jpeg
		if(jpeg_read_header(&cinfo, TRUE) != 1){
			jpeg_destroy_decompress(&cinfo);
			throw SHARKEXCEPTION("[readJPEG] Image does not look like a JPEG");
		}
		
		jpeg_start_decompress(&cinfo);
		//get image geometry and allocate space for the decompressed image
		std::size_t imgWidth =  cinfo.output_width;
		std::size_t imgHeight = cinfo.output_height;
		std::size_t channels = cinfo.output_components;
		std::vector<unsigned char> pixBuf(imgWidth * imgHeight * channels);
		
	
		//decompress image block by block
		std::vector<unsigned char*> scanlines(cinfo.rec_outbuf_height);
		while (cinfo.output_scanline < cinfo.output_height) {
			//set up the beginning of the lines to read
			for(int i = 0; i != cinfo.rec_outbuf_height; ++i){
				scanlines[i] = pixBuf.data() + (cinfo.output_scanline + i) * imgWidth * channels;
			}
			jpeg_read_scanlines(&cinfo, scanlines.data(), cinfo.rec_outbuf_height);
		}
	
		//clean-up
		jpeg_finish_decompress(&cinfo);
		jpeg_destroy_decompress(&cinfo);
		
		
		//convert to float vector
		T conv = (T)255.0;
		blas::vector<T> image(imgWidth * imgHeight * channels);
		for (std::size_t i=0; i != image.size(); ++i){
			image[i] = pixBuf[i] / conv;
		}
		
		//done
		return {std::move(image), {imgHeight, imgWidth, channels}};
	
	}catch(jpeg_error_mgr* mgr){//on failure, write message and rethrow
		char messageBuffer[JMSG_LENGTH_MAX];
		( *(mgr->format_message) ) ((j_common_ptr)&cinfo, messageBuffer);
		jpeg_abort_decompress(&cinfo);
		jpeg_destroy_decompress(&cinfo);
		throw SHARKEXCEPTION(std::string("[readJPEG] Error during reading:") + messageBuffer);
	}
}


template<class T>
std::pair<blas::vector<T>, Shape> image::readPGM(std::vector<unsigned char> const& data){
	//the header is in text format, so lets read it
	if(data[0] != 'P' || data[1] != '5')
		throw SHARKEXCEPTION("File does not have PGM format");
	
	auto pos = (char*) data.data() + 2;
	auto end = (char*) data.data() + data.size();
	while(std::isspace(*pos)) ++pos;
	//skip comments and white space
	while( pos != end && *pos == '#'){
		while(pos != end &&  *pos != '\n') ++pos;
		while(pos != end &&  std::isspace(*pos)) ++pos;
	}
	if(pos == end)
		throw SHARKEXCEPTION("Error reading header!");
	
	//read width
	long imgWidth = std::strtol(pos, &pos, 0);
	if(pos == end || imgWidth == 0)
		throw SHARKEXCEPTION("Error reading header!");
	++pos; //skip white space
	
	//read height
	long imgHeight = std::strtol(pos, &pos, 0);
	if(pos == end || imgHeight == 0)
		throw SHARKEXCEPTION("Error reading header!");
	++pos; //skip white space
	
	//read number of GrayValues
	long nGrayValues = std::strtol(pos, &pos, 0);
	if(pos == end || nGrayValues == 0)
		throw SHARKEXCEPTION("Error reading header!");
	++pos; //skip white space
	
	if(nGrayValues > 255)
		throw SHARKEXCEPTION("Only 8-Bit PGMs are supported");
	
	if( imgWidth * imgHeight > end - pos)
		throw SHARKEXCEPTION("Error: error reading image contents");
	
	
	T conv = (T)(nGrayValues +1);
	blas::vector<T> image(imgWidth * imgHeight);
	for (std::size_t i=0; i != image.size(); ++i, ++pos){
		image[i] = ((unsigned char)*pos) / conv;
	}

	return {std::move(image), {(std::size_t) imgHeight,(std::size_t) imgWidth, 1}};
}

template<class T>
std::pair<blas::vector<T>, Shape> image::readImage(std::vector<unsigned char> const& data){
	if(data[0] == 0xFF && data[1] == 0xD8)
		return readJPEG<T>(data);
	if(data[0] == 'P' && data[1] == '5')
		return readPGM<T>(data);
	if(png_sig_cmp((unsigned char*)data.data(), 0, 8) == 0)
		return readPNG<T>(data);
	throw SHARKEXCEPTION("[readImage] Could not determine image file type");
}

template<class T>
std::pair<blas::vector<T>, Shape> image::readImageFromFile(std::string const& filename){
	std::ifstream file(filename, std::ios::binary );
	if(!file)
		throw SHARKEXCEPTION("readImageFromFile] Could not open file: "+filename);
	file.seekg(0, std::ios::end);
	std::streampos fileSize = file.tellg();
	file.seekg(0, std::ios::beg);
	std::vector<unsigned char> buffer(fileSize);
	file.read((char*) &buffer[0], fileSize);
	return readImage<T>(buffer);
}


//explicit instantiation
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<float>, Shape> image::readPNG<float>(std::vector<unsigned char> const&);
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<double>, Shape> image::readPNG<double>(std::vector<unsigned char> const&);
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<float>, Shape> image::readJPEG<float>(std::vector<unsigned char> const&);
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<double>, Shape> image::readJPEG<double>(std::vector<unsigned char> const&);
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<float>, Shape> image::readPGM<float>(std::vector<unsigned char> const&);
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<double>, Shape> image::readPGM<double>(std::vector<unsigned char> const&);
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<float>, Shape> image::readImage<float>(std::vector<unsigned char> const&);
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<double>, Shape> image::readImage<double>(std::vector<unsigned char> const&);
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<float>, Shape> image::readImageFromFile<float>(std::string const&);
template SHARK_EXPORT_SYMBOL std::pair<blas::vector<double>, Shape> image::readImageFromFile<double>(std::string const&);