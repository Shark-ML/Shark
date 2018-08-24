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
	
	//we parallelize over images*rows of the result, i.e. each function evaluation computes a full row of the image
	auto resizeRow = [&](std::size_t id){
		std::size_t im = id / heightOut; //image-index
		std::size_t i = id % heightOut; //row-index in image
		auto pixelsIn = to_matrix(row(images,im), heightIn * widthIn, numChannels);
		auto pixelsOut = to_matrix(row(resizedImages,im), heightOut * widthOut, numChannels);
		
		double in_per_outi = double(heightIn)/heightOut;
		double in_per_outj = double(widthIn)/widthOut;
		
		for(std::size_t j = 0; j != widthOut; ++j){
			auto outpx = row(pixelsOut, i * widthOut + j);
			//find lower left corner of input
			double xi = i * in_per_outi;
			double xj = j * in_per_outj;
			std::size_t i_lower = std::size_t(xi);
			std::size_t j_lower = std::size_t(xj);
			//find delta values of relative pixel positions in [0,1] coordinates 
			T deltai = T(xi - i_lower);
			T deltaj = T(xj - j_lower);
			
			//lower left corner
			auto inpx_ll = row(pixelsIn, i_lower * widthIn + j_lower);
			noalias(outpx) = T(1 - deltai) * T(1 - deltaj) * inpx_ll;
			
			//lower right corner
			if(j_lower + 1 < widthIn){
				auto inpx_lr = row(pixelsIn, i_lower * widthIn + j_lower + 1);
				noalias(outpx) += T(1 - deltai) * deltaj * inpx_lr;
			}
			//upper left corner
			if(i_lower + 1 < heightIn){
				auto inpx_ur = row(pixelsIn, (i_lower+1) * widthIn + j_lower);
				noalias(outpx) += deltai * T(1 - deltaj) * inpx_ur;
			}
			
			//upper right corner
			if(j_lower + 1 < widthIn && i_lower + 1 < heightIn){
				auto inpx_ur = row(pixelsIn, (i_lower+1) * widthIn + j_lower + 1);
				noalias(outpx) += deltai * deltaj * inpx_ur;
			}
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
		
		for(std::size_t j = 0; j != widthIn; ++j){
			auto derivp = row(derivs, i * widthIn + j);
			
			double xi = i * out_per_ini;
			double xj = j * out_per_inj;
			//lower and upper bound for output pixels being affected by the input
			double mxi = (i - 1.0) * out_per_ini;
			if(mxi == std::floor(mxi))
				mxi += 1.0;
			double Mxi = (i + 1.0) * out_per_ini;
			if(Mxi == std::floor(Mxi))
				Mxi -= 1.0;
			double mxj = (j - 1.0) * out_per_inj;
			if(mxj == std::floor(mxj))
				mxj += 1.0;
			double Mxj = (j + 1.0) * out_per_inj;
			if(Mxj == std::floor(Mxj))
				Mxj -= 1.0;
			//find bounds for output pixels that were affected by the input pixel
			std::size_t imin = std::size_t(std::max(std::ceil(mxi),0.0));
			std::size_t imax = std::min(std::size_t(Mxi) + 1, heightOut);
			
			std::size_t jmin = std::size_t(std::max(std::ceil(mxj),0.0));
			std::size_t jmax = std::min(std::size_t(Mxj) + 1, heightOut);
			
			//~ if(im == 0 && i == 6){
				//~ std::cout<<out_per_ini<<" "<<out_per_inj<<std::endl;
				//~ std::cout<<j<<" "<<jmin<<" "<<jmax<<std::endl;
			//~ }
			
			//backpropagate coeffcients from affected pixel
			for(std::size_t i0 = imin; i0 < imax; ++i0){
				T deltai = T(i0 * in_per_outi - i);
				deltai = T(1) - std::abs(deltai);
				for(std::size_t j0 = jmin; j0 < jmax; ++j0){
					T deltaj = T(j0 * in_per_outj - j);
					deltaj = T(1) - std::abs(deltaj);
					//~ if(im == 0 && i == 6){
						//~ std::cout<<deltaj<<std::endl;
					//~ }
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