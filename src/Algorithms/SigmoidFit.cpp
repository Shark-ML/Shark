//===========================================================================
/*!
 * 
 *
 * \brief       Optimization of the SigmoidModel according to Platt, 1999
 * 
 * 
 *
 * \author      T. Glasmachers, O.Krause
 * \date        2010
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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


#include <shark/Algorithms/Trainers/SigmoidFit.h>
#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>

using namespace shark;

SigmoidFitRpropNLL::SigmoidFitRpropNLL( unsigned int iters ){
	m_iterations = iters;
}

// optimize the sigmoid using rprop on the negative log-likelihood
void SigmoidFitRpropNLL::train(SigmoidModel& model, LabeledData<RealVector, unsigned int> const& dataset)
{
	LinearModel<> trainModel;
	trainModel.setStructure(1,1,model.hasOffset());
	CrossEntropy loss;
	ErrorFunction <RealVector, unsigned int> modeling_error( dataset, &trainModel, &loss );
	IRpropPlus rprop;
	rprop.init( modeling_error );
	for (unsigned int i=0; i<m_iterations; i++) {
		rprop.step( modeling_error );
	}
	RealVector solution = rprop.solution().point;
	if(model.slopeIsExpEncoded()){
		solution(0) = std::log(solution(0));
	}
	if(model.hasOffset())
		solution(1) *=-1;
	model.setParameterVector(solution);
}

////////////////////////////////////////////////////////////////////////////////

// optimize the sigmoid using platt's method
void SigmoidFitPlatt::train(SigmoidModel& model, LabeledData<RealVector, unsigned int> const& dataset){
	SIZE_CHECK( model.numberOfParameters() == 2 );
	typedef LabeledData<RealVector, unsigned int>::const_element_range Elements;
	typedef IndexedIterator<boost::range_iterator<Elements>::type> Iterator;
	
	
	double a, b, c, d, e, d1, d2;
	double t = 0.0;
	double oldA, oldB, diff, scale, det;
	double err = 0.0;
	double value, p;
	double lambda = 0.001;
	double olderr = 1e100;
	std::vector<std::size_t> classCount = classSizes(dataset); 
	std::size_t pos = classCount[1];
	std::size_t neg = classCount[0];
	std::size_t ic = pos+neg;
	Iterator end(dataset.elements().end(),ic);

	double A = 0.0;
	double B = std::log((neg + 1.0) / (pos + 1.0));
	double lowTarget = 1.0 / (neg + 2.0);
	double highTarget = (pos + 1.0) / (pos + 2.0);
	RealVector pp(ic,(pos + 1.0) / (pos + neg + 2.0));
	int count = 0;
	for (std::size_t it = 0; it < 100; it++)
	{
		a = b = c = d = e = 0.0;
		
		for (Iterator it(dataset.elements().begin(),0); it != end; ++it){
			std::size_t i = it.index(); 
			t = (it->label == 1) ? highTarget : lowTarget;
			d1 = pp(i) - t;
			d2 = pp(i) * (1.0 - pp(i));
			value = it->input(0);
			a += d2 * value * value;
			b += d2;
			c += d2 * value;
			d += d1 * value;
			e += d1;
		}
		if (std::abs(d) < 1e-9 && std::abs(e) < 1e-9) break;
		oldA = A;
		oldB = B;
		err = 0.0;
		while (true)
		{
			det = (a + lambda) * (b + lambda) - c * c;
			if (det == 0.0)
			{
				lambda *= 10.0;
				continue;
			}
			A = oldA + ((b + lambda) * d - c * e) / det;
			B = oldB + ((a + lambda) * e - c * d) / det;
			err = 0.0;
			for (Iterator it(dataset.elements().begin(),0); it != end; ++it){
				std::size_t i = it.index();
				p = 1.0 / (1.0 + std::exp(A * it->input(0) + B));
				pp(i) = p;
				err -= t * safeLog(p) + (1.0 - t) * safeLog(1.0 - p);
			}
			if (err < 1.0000001 * olderr)
			{
				lambda *= 0.1;
				break;
			}
			lambda *= 10.0;
			if (lambda >= 1e6)
			{
				// Something is broken. Give up.
				break;
			}
		}
		diff = err - olderr;
		scale = 0.5 * (err + olderr + 1.0);
		if (diff > -1e-3 * scale && diff < 1e-7 * scale)
			count++;
		else
			count = 0;
		olderr = err;
		if (count == 3) break;
	}
	RealVector params(2);
	params(0) = A;
	params(1) = B;
	model.setParameterVector(params);
}
