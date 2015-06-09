//===========================================================================
/*!
 * 
 *
 * \brief       ROC
 * 
 * 
 *
 * \author      O.Krause
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ROC_H
#define SHARK_OBJECTIVEFUNCTIONS_ROC_H

#include <shark/Core/DLLSupport.h>
#include <shark/Models/AbstractModel.h>
#include <shark/Data/Dataset.h>
#include <vector>
#include <algorithm>

namespace shark {

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
//! be the labels 0 and 1 of a binary classification task. The
//! model output is assumed not to be 0 and 1, but real valued
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
	//! \param  set     data set with inputs and corresponding binary outputs (0 or 1)
	template<class InputType>
	ROC(AbstractModel<InputType,RealVector>& model,LabeledData<InputType,unsigned int> const& set){
		std::size_t inputs=set.numberOfElements();

		//calculat the number of classes
		std::vector<std::size_t> classes = classSizes(set);
		SIZE_CHECK(classes.size() == 2); //only binary problems allowed!
		
		std::size_t positive = classes[0];
		std::size_t negative = classes[1];
		m_scorePositive.resize(positive);
		m_scoreNegative.resize(negative);

		// compute scores
		std::size_t posPositive = 0;
		std::size_t posNegative = 0;
		
		//calculate the model responses batchwise for the whole set
		for(std::size_t i = 0; i != set.size(); ++i){
			RealMatrix output = model(set.batch(i).input);
			SIZE_CHECK(output.size2() == 1);
			for(std::size_t j = 0; j != size(output); ++j){ 
				double value = output(j,0);
				if (set.batch(i)(j) == 1)
				{
					m_scorePositive[posPositive] = value;
					posPositive++;
				}
				else
				{
					m_scoreNegative[posNegative] = value;
					posNegative++;
				}
			}
		}
		// sort positives and negatives by score
		std::sort(m_scorePositive.begin(), m_scorePositive.end());
		std::sort(m_scoreNegative.begin(), m_scoreNegative.end());
	}

	//! Compute the threshold for given false acceptance rate,
	//! that is, for a given false positive rate.
	//! This threshold, used for classification with the underlying
	//! model, results in the given false acceptance rate.
	SHARK_EXPORT_SYMBOL double threshold(double falseAcceptanceRate)const;

	//! Value of the ROC curve for given false acceptance rate,
	//! that is, for a given false positive rate.
	SHARK_EXPORT_SYMBOL double value(double falseAcceptanceRate)const;

	//! Computes the equal error rate of the classifier
	SHARK_EXPORT_SYMBOL double equalErrorRate()const;

protected:
	//! scores of the positive examples
	std::vector<double> m_scorePositive;

	//! scores of the negative examples
	std::vector<double> m_scoreNegative;
};

}
#endif
