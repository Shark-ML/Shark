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
#include <vector>
#include <string>

namespace shark { namespace image{
	
/// \brief Encodes an image as PNG
/// 
/// PNG supports all Pixel Types. The image shape is interpreted as height x width x channels, where the number
/// of channels must match the channels indicated by type (e.g. RGB requires exactly 3 channels, Luma 1)
template<class T>
std::vector<unsigned char> writePNG(blas::vector<T> const& image, Shape const& shape, PixelType type);

/// \brief Encodes an image as JPEG
/// 
/// JPEG does not support images with alpha channels. The image shape is interpreted as height x width x channels, where the number
/// of channels must match the channels indicated by type (e.g. RGB requires exactly 3 channels, Luma 1)
template<class T>
std::vector<unsigned char> writeJPEG(blas::vector<T> const& image, Shape const& shape, PixelType type);

/// \brief Encodes LUMA images as PNG
/// 
/// PGM only supports single channel Luma images. The image shape is interpreted as height x width x 1. 
template<class T>
std::vector<unsigned char> writePGM(blas::vector<T> const& image, Shape const& shape, PixelType type);

/// \brief Stores an image to a file
///
/// The image format is read from the extension, e.g. ".png" calls writePNG. Images are always encoded with 8 bit per pixel per channel.
/// exceptions are thrown if the indicated pixel type can not be stored by the image format (e.g. alpha channels in jpeg).
template<class T>
void writeImageToFile(std::string const& filename, blas::vector<T> const& image, Shape const& shape, PixelType type);

	
//implementation in cpp for float and double cpu vectors only
extern template std::vector<unsigned char> writePNG<float>(blas::vector<float> const& image, Shape const& shape, PixelType type);
extern template std::vector<unsigned char> writePNG<double>(blas::vector<double> const& image, Shape const& shape, PixelType type);
extern template std::vector<unsigned char> writeJPEG<float>(blas::vector<float> const& image, Shape const& shape, PixelType type);
extern template std::vector<unsigned char> writeJPEG<double>(blas::vector<double> const& image, Shape const& shape, PixelType type);
extern template std::vector<unsigned char> writePGM<float>(blas::vector<float> const& image, Shape const& shape, PixelType type);
extern template std::vector<unsigned char> writePGM<double>(blas::vector<double> const& image, Shape const& shape, PixelType type);
extern template void writeImageToFile(std::string const& filename, blas::vector<float> const& image, Shape const& shape, PixelType type);
extern template void writeImageToFile(std::string const& filename, blas::vector<double> const& image, Shape const& shape, PixelType type);


}}
#endif
