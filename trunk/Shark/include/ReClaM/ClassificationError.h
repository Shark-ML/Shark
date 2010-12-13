//===========================================================================
/*!
*  \file ClassificationError.h
*
*  \brief Compute the fraction of classification errors
*
*  \author  T. Glasmachers
*  \date    2006
*
*
* \par
* Both the #ClassificationError and the #BalancedClassificationError
* assume one dimensional model output and target values. There are
* two variants, that is, the output and target arrays may be one
* dimensional with the dimension representing the data points, or
* the arrays may contain an exclicit second dimension. In the second
* case the output dimension must equal 1. In all cases there has to
* be one real valued output and target per input pattern.
* This covers both the cases of support vector machines and neural
* networks with a single output neuron. For multi class neural
* networks refer the the winner takes all class.
*
* \par
* It is assumed that the classication is done by comparison with a
* threshold, usually zero. This includes the special case of taking
* the sign function for classification.
* For neural networks with sigmoidal transfer function of the output
* neuron it is usually assumed that the classification boundary is
* the value \f$ 0.5 \f$. Therefore it is possible to provide a
* nonzero threshold parameter at construction time.
*
*
*  \par Copyright (c) 1999-2006:
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
*
*
*  <BR><HR>  
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
*/
//===========================================================================


#ifndef _ClassificationError_H_
#define _ClassificationError_H_


#include <ReClaM/ErrorFunction.h>


//! The ClassificationError class returns the number of
//! classification errors. This measure is not differentiable.
class ClassificationError : public ErrorFunction
{
public:
	//! Constructor
	ClassificationError(double threshold = 0.0);

	//! Destructor
	~ClassificationError();


	//! Computation of the fraction of wrongly classified examples.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! return the false positive rate
	//! of the last error evaluation
	inline double falsePositiveRate()
	{
		return fpr;
	}

	//! return the false negative rate
	//! of the last error evaluation
	inline double falseNegativeRate()
	{
		return fnr;
	}

	//! return the true positive rate
	//! of the last error evaluation
	inline double truePositiveRate()
	{
		return 1. - fnr;
	}


protected:
	double threshold;
	double fpr;
	double fnr;
};


//! The ClassificationError class returns the number of
//! classification errors, rescaled by the class magnitudes.
//! For unbalanced datasets, it is in most cases preferable
//! compared to the #ClassificationError.
class BalancedClassificationError : public ErrorFunction
{
public:
	//! Constructor
	BalancedClassificationError(double threshold = 0.0);

	//! Destructor
	~BalancedClassificationError();


	//! Computation of the fraction of wrongly classified examples,
	//! rescaled as follows:
	//! let p be the number of positive class examples and let
	//! n be the number of negative class examples. Let a be the
	//! number of positive examples classified negative (type 1 error)
	//! and let b be the number of negative examples classifies positive
	//! (type 2 error). Then, the balanced classification error is the
	//! error rate
	//! \f[
	//!       \frac{1}{2} \left( \frac{a}{p} + \frac{b}{n} \right) .
	//! \f]
	//! The expectation value of the balanced classification error is
	//! invariant under the class magnitude, that is the fraction of
	//! positive and negative class examples. For example, if all
	//! positive examples are doubled in the test set, the error rate
	//! remains unchanged.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

protected:
	double threshold;
};


//! The 0-1-loss counts the number of errors,
//! where any deviation of the prediction from
//! the target counts as one error, while only
//! an exact match counts as a correct prediction.
//! E.g., zhis error measure is useful for multi
//! class SVMs.
class ZeroOneLoss : public ErrorFunction
{
public:
	//! Constructor
	ZeroOneLoss();

	//! Destructor
	~ZeroOneLoss();


	//! Computation of the fraction of wrongly classified examples.
	double error(Model& model, const Array<double>& input, const Array<double>& target);
};


#endif

