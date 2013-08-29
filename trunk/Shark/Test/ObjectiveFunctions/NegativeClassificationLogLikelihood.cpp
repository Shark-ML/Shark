//===========================================================================
/*!
 *  \file NegativeClassificationLogLikelihood.cpp
 *
 *  \brief NegativeClassificationLogLikelihood test case
 *
 *
 *  \author M.Tuma
 *  \date 2011
 *
 *  \par Copyright (c) 1998-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#include <shark/ObjectiveFunctions/Loss/NegativeClassificationLogLikelihood.h>
#include <cmath>

#include "TestLoss.h"

#define BOOST_TEST_MODULE OBJECTIVEFUNCTIONS_NEGATIVECLASSIFICATIONLOGLIKELIHOOD
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( NCLL_EVAL_BINARY ) {
	NegativeClassificationLogLikelihood ncll;
	RealMatrix input(4,1);
	input(0,0)=0.8;
	input(1,0)=0.2;
	input(2,0)=0.9;
	input(3,0)=0.3;
	UIntVector label(4);
	label(0)=1;
	label(1)=1;
	label(2)=0;
	label(3)=0;
	
	// first, some manual tests for binary
	BOOST_CHECK_SMALL( ncll.eval(blas::repeat(1,1),blas::repeat(0.8,1,1))-0.223143551 , 1e-7);
	BOOST_CHECK_SMALL( ncll.eval(blas::repeat(1,1),blas::repeat(0.2,1,1))-1.609437912 , 1e-7);
	BOOST_CHECK_SMALL( ncll.eval(blas::repeat(0,1),blas::repeat(0.9,1,1))-2.302585093 , 1e-7);
	BOOST_CHECK_SMALL( ncll.eval(blas::repeat(0,1),blas::repeat(0.3,1,1))-0.356674944 , 1e-7);
	BOOST_CHECK_SMALL( ncll.eval(blas::repeat(0,1),blas::repeat(0.5,1,1))-ncll.eval(blas::repeat(1,1),blas::repeat(0.5,1,1)), 1e-12);
	// second, same for binary derivatives
	RealMatrix deriv, deriv2;
	BOOST_CHECK_SMALL( ncll.evalDerivative(blas::repeat(1,1),blas::repeat(0.8,1,1), deriv)-0.223143551 , 1e-7);
	BOOST_CHECK_SMALL( deriv(0,0) + 1.25, 1e-7);
	BOOST_CHECK_SMALL( ncll.evalDerivative(blas::repeat(1,1),blas::repeat(0.2,1,1), deriv)-1.609437912 , 1e-7);
	BOOST_CHECK_SMALL( deriv(0,0) + 5.0, 1e-7);
	BOOST_CHECK_SMALL( ncll.evalDerivative(blas::repeat(0,1),blas::repeat(0.9,1,1), deriv)-2.302585093 , 1e-7);
	BOOST_CHECK_SMALL( deriv(0,0) - 10.0, 1e-7);
	BOOST_CHECK_SMALL( ncll.evalDerivative(blas::repeat(0,1),blas::repeat(0.3,1,1), deriv)-0.356674944 , 1e-7);
	BOOST_CHECK_SMALL( deriv(0,0) - 1.4285714286, 1e-7);
	BOOST_CHECK_SMALL( ncll.evalDerivative(blas::repeat(0,1),blas::repeat(0.5,1,1), deriv)-ncll.evalDerivative(blas::repeat(1,1),blas::repeat(0.5,1,1),deriv2), 1e-12);
	BOOST_CHECK_SMALL( deriv(0,0) - 2.0, 1e-7);
	
	//third, now for a whole batch
	const double result=0.223143551+1.609437912+2.302585093+0.356674944;
	BOOST_CHECK_SMALL( ncll.eval(label,input)-result , 1e-7);
	BOOST_CHECK_SMALL( ncll.evalDerivative(label,input,deriv)-result , 1e-7);
	BOOST_CHECK_SMALL( deriv(0,0) + 1.25, 1e-7);
	BOOST_CHECK_SMALL( deriv(1,0) + 5.0, 1e-7);
	BOOST_CHECK_SMALL( deriv(2,0) - 10.0, 1e-7);
	BOOST_CHECK_SMALL( deriv(3,0) - 1.4285714286, 1e-7);
}
// then, automatically test multi-class variant
BOOST_AUTO_TEST_CASE( NCLL_EVAL_MULTI_CLASS ) {
	NegativeClassificationLogLikelihood ncll;
	unsigned int maxTests = 500;
	for (unsigned int test = 0; test != maxTests; ++test)
	{
		unsigned int d, dim = Rng::discrete(5, 100);
		UIntVector label(1);
		RealMatrix probs(1,dim);
		for (d=0; d<dim; d++)
			probs(0,d) = Rng::uni(0.01, 1.0);
		label(0) = Rng::discrete(0, dim-1);
		double l = ncll.eval(label, probs);
		double check = -std::log(probs(0,label(0)));
		BOOST_CHECK_SMALL(l - check, 1e-12);
		// test first evalDerivative (automagically via TestLoss.h)
		RealMatrix derivative;
		double value = ncll.evalDerivative(label, probs, derivative);
		RealVector estimatedDerivative = estimateDerivative(ncll, probs, label);
		//std::cout<<derivative<< " " << estimatedDerivative<<std::endl;
		//std::cout<<probs << " "<<label<<std::endl;
		BOOST_CHECK_SMALL(value - check, 1.e-13);
		BOOST_CHECK_SMALL(norm_2(row(derivative,0) - estimatedDerivative), 1.e-4);
		
		//now a whole batch just by copying
		RealMatrix probsBatch(10,dim);
		UIntVector labelBatch(10);
		for(std::size_t i = 0; i != 10; ++i){
			row(probsBatch,i)=row(probs,0);
			labelBatch(i)=label(0);
		}
		double valueBatch = ncll.evalDerivative(labelBatch, probsBatch, derivative);
		BOOST_CHECK_SMALL(valueBatch - 10*check, 1.e-12);
		for(std::size_t i = 0; i != 10; ++i){
			BOOST_CHECK_SMALL(norm_2(row(derivative,i) - estimatedDerivative), 1.e-4);
		}
	}
}
