//===========================================================================
/*!
*  \file NormedModels.h
*
*  \brief Wrapper models normalizing the output.
*
*  \author  C. Igel
*  \date    2009
*
*  \par Copyright (c)
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*  \par Project:
*      ReClaM
*
*
*  <BR>
*  This file is part of ReClaM. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 2, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, write to the Free Software
*  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*
*
*/
//===========================================================================

#ifndef _NORMED_MODELS_H_
#define _NORMED_MODELS_H_


#include <ReClaM/Model.h>
#include <ReClaM/RException.h>
#include <Array/ArrayOp.h>

/*!
 *  \brief Normalizes the output of a base model to sum to one using softmax.
 */
class ExpNorm : public Model
{
public:
  ExpNorm(Model *m) {
    
    m_baseModel = m;

    epsilon = 1E-10;

    inputDimension = m_baseModel->getInputDimension();
    outputDimension = m_baseModel->getOutputDimension();
    
    parameter.resize(m_baseModel->getParameterDimension(), false);
    for (unsigned i = 0; i < getParameterDimension(); i++) 
      parameter(i) = m_baseModel->getParameter(i);
  }
  
  ~ExpNorm() {
  }


  void model(const Array<double>& input, Array<double> &output)
  {
    double sum;
    
    m_baseModel->model(input, output);
    if (input.ndim() == 1) {
      sum = 0.;
      for (unsigned i = 0; i < outputDimension; i++) {
				output(i) = exp(output(i));
				sum += output(i);
      }
      if(!sum) throw REXCEPTION("[ExpNorm::model] 1st branch, zero denominator");
      if(!finite(sum)) throw REXCEPTION("[ExpNorm::model] 1st branch, denominator not finite");
      output /= sum;
    }
    else if (input.ndim() == 2) {
      unsigned jc;
      jc = input.dim(0);
      output.resize(jc, outputDimension, false);
      for (unsigned j = 0; j < jc; j++) { // loop over patterns
				sum = 0.;
				for (unsigned i = 0; i < outputDimension; i++) {
					output(j, i) = exp(output(j, i));
					sum += output(j, i);
				}
				if(!sum) REXCEPTION("[ExpNorm::model] zero denominator");
				if(!finite(sum)) REXCEPTION("[ExpNorm::model] denominator not finite");
				for (unsigned i = 0; i < outputDimension; i++) output(j, i) /= sum;
      }
    }
  }
	
  void modelDerivative(const Array<double>& input, Array<double>& derivative)
  {
    Array<double> output, baseDerivative;
    model(input, output);
    derivative    .resize(outputDimension, getParameterDimension());
    baseDerivative.resize(outputDimension, getParameterDimension());


    baseDerivative = 0;
    m_baseModel->modelDerivative(input, baseDerivative);


		for (unsigned i = 0; i < outputDimension; i++) {
			for (unsigned j = 0; j < derivative.dim(1); j++) {
				double sum = 0;
				for (unsigned k = 0; k < outputDimension; k++) sum += output(k) * baseDerivative(k, j);
				derivative(i, j) = baseDerivative(i, j) * output(i) - output(i) * sum;
				if(!finite(derivative(i, j))) throw REXCEPTION("[ExpNorm::modelDerivative] derivative not finite");
      }
    }
  }

  void setParameter(unsigned int index, double value)
  {
    parameter(index) = value;
    m_baseModel->setParameter(index, value);
  }

 protected: 

  Model *m_baseModel;
};

/*!
 *  \brief Normalizes the (non-negative) output of a base model by dividing by the overall sum.
 */
class LinNorm : public Model
{
public:
  LinNorm(Model *m) {
    
    m_baseModel = m;

    epsilon = 1E-10;

    inputDimension = m_baseModel->getInputDimension();
    outputDimension = m_baseModel->getOutputDimension();
    
    parameter.resize(m_baseModel->getParameterDimension(), false);
    for (unsigned i = 0; i < getParameterDimension(); i++) 
      parameter(i) = m_baseModel->getParameter(i);
  }
  
  ~LinNorm() {
  }

  void model(const Array<double>& input, Array<double> &output)
  {
    double sum;
    
    m_baseModel->model(input, output);
    if (input.ndim() == 1) {
      sum = 0.;
      for (unsigned i = 0; i < outputDimension; i++) {
				sum += output(i);
      }
      if(!sum) throw REXCEPTION("[ExpNorm::model] 1st branch, zero denominator");
      if(!finite(sum)) throw REXCEPTION("[ExpNorm::model] 1st branch, denominator not finite");
      output /= sum;
    }
    else if (input.ndim() == 2) {
      unsigned jc;
      jc = input.dim(0);
      output.resize(jc, outputDimension, false);
      for (unsigned j = 0; j < jc; j++) { // loop over patterns
				sum = 0.;
				for (unsigned i = 0; i < outputDimension; i++) {
					sum += output(j, i);
				}
				if(!sum) REXCEPTION("[ExpNorm::model] zero denominator");
				if(!finite(sum)) REXCEPTION("[ExpNorm::model] denominator not finite");
				for (unsigned i = 0; i < outputDimension; i++) output(j, i) /= sum;
      }
    }
  }
	
  void modelDerivative(const Array<double>& input, Array<double>& derivative)
  {
    Array<double> output, baseDerivative;

    // in contrast to ExpNorm, we need the output of the base model
    m_baseModel->model(input, output);
    derivative    .resize(outputDimension, getParameterDimension());
    baseDerivative.resize(outputDimension, getParameterDimension());


    baseDerivative = 0;
    m_baseModel->modelDerivative(input, baseDerivative);

    double outputSum = 0.;
    for (unsigned i = 0; i < outputDimension; i++) {
      outputSum += output(i);
    }

    for (unsigned i = 0; i < outputDimension; i++) {
      for (unsigned j = 0; j < derivative.dim(1); j++) {
				double sum = 0;
				for (unsigned k = 0; k < outputDimension; k++) sum +=  baseDerivative(k, j);
				derivative(i, j) = (baseDerivative(i, j) - (output(i) / outputSum) * sum) / outputSum;
				if(!finite(derivative(i, j))) throw REXCEPTION("[ExpNorm::modelDerivative] derivative not finite");
      }
    }

  }

  void setParameter(unsigned int index, double value)
  {
    parameter(index) = value;
    m_baseModel->setParameter(index, value);
  }

 protected: 

  Model *m_baseModel;
};
#endif
