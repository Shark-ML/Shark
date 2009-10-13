//===========================================================================
/*!
 *  \file ROC.h
 *
 *  \brief computes a "receiver operator characteristics" curve
 *
 *  \author  T. Glasmachers
 *  \date    2006
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


#ifndef _ROC_H_
#define _ROC_H_


#include <ReClaM/Model.h>
#include <vector>


//!
//! \brief ROC-Curve - false negatives over false positives
//!
//! \par
//! This class provides the ROC curve of a classifier.
//! All time consuming computations are done in the constructor,
//! such that afterwards fast access to specific values of the
//! curve and the equal error rate is possible.
//!
//! \par
//! The ROC class assumes a one dimensional target array and a
//! model producing one dimensional output data. The targets must
//! be the labels +1 and -1 of a binary classification task. The
//! model output is assumed not to be +1 and -1, but real valued
//! instead. Classification in done by thresholding, where
//! different false positive and false negative rates correspond
//! to different thresholds. The ROC curve shows the trade off
//! between the two error types.
//!
class ROC
{
public:
	//! Constructor
	//!
	//! \param  model   model to use for prediction
	//! \param  input   input data
	//! \param  target  binary targets for the input data (+1 or -1)
	ROC(Model& model, const Array<double>& input, const Array<double>& target);

	//! Compute the threshold for given false acceptance rate,
	//! that is, for a given false positive rate.
	//! This threshold, used for classification with the underlying
	//! model, results in the given false acceptance rate.
	double Threshold(double falseAcceptanceRate);

	//! Value of the ROC curve for given false acceptance rate,
	//! that is, for a given false positive rate.
	double Value(double falseAcceptanceRate);

	//! Computes the equal error rate of the classifier
	double EqualErrorRate();

protected:
	//! scores of the positive examples
	std::vector<double> score_positive;

	//! scores of the negative examples
	std::vector<double> score_negative;
};


#endif

