//===========================================================================
/*!
 * 
 *
 * \brief       Importing and exporting PGM images
 * 
 * 
 *
 * \author      C. Igel
 * \date        2011
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

#ifndef SHARK_DATA_LOAD_IMAGE_H
#define SHARK_DATA_LOAD_IMAGE_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
#include <memory>

extern "C"{
#include <png.h>
}


namespace shark { namespace image{
	
namespace detail{
//set up the data streaming callback
//we wrap our data in a source that is fed to the callback
struct pngIOSource{
	unsigned char const* data;
	std::size_t pos;
};
//callback used by libpng to read the next chunk of data
void readPNGData(png_struct* pngPtr, unsigned char* data, std::size_t length) {
	pngIOSource* source = (pngIOSource*)png_get_io_ptr(pngPtr);
	for(std::size_t i = 0; i != length; ++i, ++source->pos){
		data[i] = source->data[source->pos];
	}
}
}
template<class T>
std::pair<blas::vector<T>, Shape> readPNG(std::vector<unsigned char> const& data){
	//Let LibPNG check the sig. If this function returns 0, everything is OK.
	if(png_sig_cmp(data.data(), 0, 8) != 0)
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

	//set up the callback
	detail::pngIOSource source ={data.data(), 0};
	png_set_read_fn(pngPtr,(void*)&source, detail::readPNGData);
	
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

}}
#endif
