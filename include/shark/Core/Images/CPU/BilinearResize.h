//===========================================================================
/*!
 * 
 *
 * \brief       Bilinear Interpolation on CPU
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


#ifndef SHARK_CORE_IMAGES_CPU_BILINEAR_RESIZE_H
#define SHARK_CORE_IMAGES_CPU_BILINEAR_RESIZE_H

#include <shark/LinAlg/Base.h>
#include <shark/Core/Shape.h>
#include <shark/Core/Threading/Algorithms.h>
namespace shark{
namespace image{
template<class T>
void bilinearResize(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> images, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> resizedImages,
	Shape const& inShape,
	Shape const& outShape
){
	SIZE_CHECK(inShape.size() == outShape.size());
	SIZE_CHECK(inShape.size() == 2 || inShape[2] == outShape[2]);
	SIZE_CHECK(images.size1() == resizedImages.size1());
	SIZE_CHECK(images.size2() == inShape.numElements());
	SIZE_CHECK(resizedImages.size2() == outShape.numElements());
	
	std::size_t heightIn = inShape[0];
	std::size_t widthIn = inShape[1];
	std::size_t heightOut = outShape[0];
	std::size_t widthOut = outShape[1];
	std::size_t numChannels = (inShape.size() == 3)? inShape[2]: 1;
	
	//fill with 0
	resizedImages.clear();
	
	//we parallelize over images*rows of the result, i.e. each function evaluation computes a full row of the image
	auto resizeRow = [&](std::size_t id){
		std::size_t im = id / heightOut; //image-index
		std::size_t i = id % heightOut; //row-index in image
		auto pixelsIn = to_matrix(row(images,im), heightIn * widthIn, numChannels);
		auto pixelsOut = to_matrix(row(resizedImages,im), heightOut * widthOut, numChannels);
		
		double in_per_outi = double(heightIn)/(heightOut);
		double in_per_outj = double(widthIn)/(widthOut);
		
		//correction in x-coordinates to correct for the change in first pixel position
		//compared to the original image.
		double basexi = 0.5 * in_per_outi - 0.5;
		double basexj = 0.5 * in_per_outj - 0.5;
		
		for(std::size_t j = 0; j != widthOut; ++j){
			auto outpx = row(pixelsOut, i * widthOut + j);
			//calculate coordinates of point wrt input pixel scale
			//we take into account that pixels have an area, therefore
			//when upsampling, we have to sample negative coordinates
			double xi = i * in_per_outi + basexi;
			double xj = j * in_per_outj + basexj;
			//find delta values of relative pixel positions in [0,1] coordinates 
			T deltai = T(xi - std::floor(xi));
			T deltaj = T(xj - std::floor(xj));
			
			//obtain the 4 corner pixels. the 1.e-8 corrects for rounding errors
			int il = int(std::floor(xi + 1.e-8));
			int iu =int(std::ceil(xi - 1.e-8));
			int jl = int(std::floor(xj + 1.e-8));
			int jr = int(std::ceil(xj - 1.e-8));
			
			//lower left corner
			if(il >= 0 && jl  >= 0)
				noalias(outpx) += T(1 - deltai) * T(1 - deltaj) * row(pixelsIn, il * widthIn + jl);
			//lower right corner
			if(il >= 0 && jr  < (int)widthIn)
				noalias(outpx) += T(1 - deltai) * deltaj * row(pixelsIn, il * widthIn + jr);
			//upper left corner
			if(iu < (int)heightIn && jl  >= 0)
				noalias(outpx) += deltai * T(1 - deltaj) * row(pixelsIn, iu * widthIn + jl);
			//upper right corner
			if(iu < (int)heightIn && jr < (int)widthIn)
				noalias(outpx) += deltai * deltaj * row(pixelsIn, iu * widthIn + jr);
		}
	};
	threading::parallelND({images.size1() * heightOut}, {0}, resizeRow, threading::globalThreadPool());
}


template<class T>
void bilinearResizeWeightedDerivative(
	blas::dense_matrix_adaptor<T const, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> coefficients, 
	blas::dense_matrix_adaptor<T, blas::row_major, blas::continuous_dense_tag, blas::cpu_tag> inputDerivatives,
	Shape const& inShape,
	Shape const& outShape
){
	SIZE_CHECK(inShape.size() == outShape.size());
	SIZE_CHECK(inShape.size() == 2 || inShape[2] == outShape[2]);
	SIZE_CHECK(coefficients.size1() == inputDerivatives.size1());
	SIZE_CHECK(coefficients.size2() == outShape.numElements());
	SIZE_CHECK(inputDerivatives.size2() == inShape.numElements());
	
	std::size_t heightIn = inShape[0];
	std::size_t widthIn = inShape[1];
	std::size_t heightOut = outShape[0];
	std::size_t widthOut = outShape[1];
	std::size_t numChannels = (inShape.size() == 3)? inShape[2]: 1;
	inputDerivatives.clear();
	
	
	//we parallelize over images*rows of the result, i.e. each function evaluation computes a full row of the image
	auto derivativeRow = [&](std::size_t id){
		std::size_t im = id / heightIn; //image-index
		std::size_t i = id % heightIn; //row-index in derivative,aka input image row
		auto coeffs = to_matrix(row(coefficients,im), heightOut * widthOut, numChannels);
		auto derivs = to_matrix(row(inputDerivatives,im), heightIn * widthIn, numChannels);
		
		double out_per_ini = double(heightOut)/heightIn;
		double out_per_inj = double(widthOut)/widthIn;
		
		double in_per_outi = 1.0/out_per_ini;
		double in_per_outj = 1.0/out_per_inj;
		
		//correction in x-coordinates to correct for the change in first pixel position between original and resized image.
		//compared to the original image. the 1.e-8 corrects for rounding errors
		double basexi = 0.5 - 0.5 * out_per_ini;
		double basexj = 0.5 - 0.5 * out_per_inj;
		
		for(std::size_t j = 0; j != widthIn; ++j){
			auto derivp = row(derivs, i * widthIn + j);
			
			double xi = i * out_per_ini - basexi;
			double xj = j * out_per_inj - basexj;
			
			//lower and upper bound for output pixels being affected by the input
			double mxi = xi - out_per_ini - 1.e-8;
			double Mxi = xi + out_per_ini + 1.e-8;
			double mxj = xj - out_per_inj - 1.e-8;
			double Mxj = xj + out_per_inj + 1.e-8;
			if(mxi == std::floor(mxi))
				mxi += 1.0;
			
			if(Mxi == std::floor(Mxi))
				Mxi -= 1.0;
			
			if(mxj == std::floor(mxj))
				mxj += 1.0;
			
			if(Mxj == std::floor(Mxj))
				Mxj -= 1.0;
			//find bounds for output pixels that were affected by the input pixel
			std::size_t imin = std::size_t(std::max(std::ceil(mxi),0.0));
			std::size_t imax = std::min(std::size_t(Mxi) + 1, heightOut);
			
			std::size_t jmin = std::size_t(std::max(std::ceil(mxj),0.0));
			std::size_t jmax = std::min(std::size_t(Mxj) + 1, widthOut);
			
			//backpropagate coeffcients from affected pixel
			for(std::size_t i0 = imin; i0 < imax; ++i0){
				T deltai = T((i0 - xi) * in_per_outi);
				deltai = T(1) - std::abs(deltai);
				for(std::size_t j0 = jmin; j0 < jmax; ++j0){
					T deltaj = T((j0 - xj) * in_per_outj);
					deltaj = T(1) - std::abs(deltaj);
					auto coeffp = row(coeffs, i0 * widthOut + j0);
					noalias(derivp) += deltai * deltaj* coeffp;
				}
			}
		}
	};
	threading::parallelND({inputDerivatives.size1() * heightIn}, {0}, derivativeRow, threading::globalThreadPool());
}


}}

#endif