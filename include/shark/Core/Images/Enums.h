//===========================================================================
/*!
 * 
 *
 * \brief       Enums for Image handling
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

#ifndef SHARK_CORE_IMAGE_ENUMS_H
#define SHARK_CORE_IMAGE_ENUMS_H

namespace shark{
enum class Interpolation{
	Linear, 
	Spline
};
enum class Padding{
	Valid,
	ZeroPad
};

enum class PixelType{
	RGB,
	RGBA,
	ARGB,
	Luma
};

enum class ImageFormat{
	NHWC = 1234,
	NCHW = 1423,
	CNHW = 4123,
};

}

#endif