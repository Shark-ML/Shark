//===========================================================================
/*!
 * 
 *
 * \brief       Importing of Images
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

#ifndef SHARK_DATA_READ_IMAGE_H
#define SHARK_DATA_READ_IMAGE_H
#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
#include <vector>
#include <utility>
#include <string>

namespace shark { namespace image{
	
/// \brief Reads a png file stored in a given vector
///
/// The returned shape must be interpreted as channels x height x width. If channel = 3 the stored image was
/// stored as RGB. If channels are 4, the image was encoded as RGBA (even if the original image is stored as ARGB).
/// a single channel is interpreted as Luma
template<class T>
std::pair<blas::vector<T>, Shape> readPNG(std::vector<unsigned char> const& data);

/// \brief Reads a jpeg file stored in a given vector
///
/// The returned shape must be interpreted as channels x height x width. If channel = 3 the stored image was
/// stored as RGB. A single channel is interpreted as Luma.
template<class T>
std::pair<blas::vector<T>, Shape> readJPEG(std::vector<unsigned char> const& data);

/// \brief Reads a pgm file stored in a given vector
///
/// The returned shape must be interpreted as channels x height x width, where channels = 1.
template<class T>
std::pair<blas::vector<T>, Shape> readPGM(std::vector<unsigned char> const& data);
/// \brief Tries to guess the file format of a file stored in data and read as image
///
/// Typically the first bytes of a file encode its file type. Supported file types are png, jpeg and pgm.
/// The returned shape must be interpreted as channels x height x width. If channel = 3 the stored image was
/// stored as RGB. If channels are 4, the image was encoded as RGBA (even if the original image is stored as ARGB).
/// a single channel is interpreted as Luma
template<class T>
std::pair<blas::vector<T>, Shape> readImage(std::vector<unsigned char> const& data);
/// \brief Reads an image file
///
/// Typically the first bytes of a file encode its file type. Supported file types are png, jpeg and pgm.
/// The returned shape must be interpreted as channels x height x width. If channel = 3 the stored image was
/// stored as RGB. If channels are 4, the image was encoded as RGBA (even if the original image is stored as ARGB).
/// a single channel is interpreted as Luma
template<class T>
std::pair<blas::vector<T>, Shape> readImageFromFile(std::string const& filename);

	
//implementation in cpp for float and double cpu vectors only
extern template std::pair<blas::vector<float>, Shape> readPNG<float>(std::vector<unsigned char> const&);
extern template std::pair<blas::vector<double>, Shape> readPNG<double>(std::vector<unsigned char> const&);
extern template std::pair<blas::vector<float>, Shape> readJPEG<float>(std::vector<unsigned char> const&);
extern template std::pair<blas::vector<double>, Shape> readJPEG<double>(std::vector<unsigned char> const&);
extern template std::pair<blas::vector<float>, Shape> readPGM<float>(std::vector<unsigned char> const&);
extern template std::pair<blas::vector<double>, Shape> readPGM<double>(std::vector<unsigned char> const&);
extern template std::pair<blas::vector<float>, Shape> readImage<float>(std::vector<unsigned char> const&);
extern template std::pair<blas::vector<double>, Shape> readImage<double>(std::vector<unsigned char> const&);
extern template std::pair<blas::vector<float>, Shape> readImageFromFile<float>(std::string const&);
extern template std::pair<blas::vector<double>, Shape> readImageFromFile<double>(std::string const&);
}}
#endif
