//===========================================================================
/*!
 * 
 * \file        NegativeAUC.h
 *
 * \brief       Functions for measuring the area under the (ROC) curve
 * 
 * 
 *
 * \author      Christian Igel
 * \date        2011
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_NEGATIVE_AUC_H
#define SHARK_OBJECTIVEFUNCTIONS_NEGATIVE_AUC_H


#include <shark/ObjectiveFunctions/AbstractCost.h>
#include <shark/Core/utility/KeyValuePair.h>

namespace shark {
///
/// \brief Negative area under the curve
/// 
/// This class computes the area under the ROC (receiver operating characteristic) curve.
/// It implements the algorithm described in:
/// Tom Fawcett. ROC Graphs: Notes and Practical Considerations for Researchers. 2004
///
/// The area is negated so that optimizing the AUC corresponds to a minimization task. 
///
template<class LabelType = unsigned int, class OutputType = RealVector>
class NegativeAUC : public AbstractCost<LabelType, OutputType>
{
 public:
	typedef KeyValuePair< double, LabelType > AUCPair;

	/// Constructor.
	/// \param invert: if set to true, the role of positive and negative class are switched
	NegativeAUC(bool invert = false) {
		m_invert = invert;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NegativeAUC"; }

	/// \brief Computes area under the curve.
  	/// \param target: class label, 0 or 1
	/// \param prediction: prediction by classifier, OutputType-valued vector
	/// \param column: indicates the column of the prediction vector interpreted as probability of positive class
	double eval(Data<LabelType> const& target, Data<OutputType> const& prediction, unsigned int column) const {
		SHARK_CHECK(dataDimension(prediction) > column,"[NegativeAUC::eval] column number too large");

		std::size_t elements = target.numberOfElements();
		
		unsigned P = 0; // positive examples
		unsigned N = 0; // negative examples
		std::vector<AUCPair> L(elements); // list of predictions and labels
		
		for(std::size_t i=0; i!= elements; i++) { // build list
			LabelType t = target.element(i);
			// negate predictions if m_invert is set
			if(!m_invert)
				L[i] = AUCPair(prediction.element(i)(column), t);
			else
				L[i] = AUCPair(-prediction.element(i)(column), t);
			// count positive and negative examples
			if(t > 0) 
				P++;
			else 
				N++;
		}    
					  
		std::sort (L.begin(), L.end(),std::greater<AUCPair>() ); // sort in decreasing order
	
		double   A = 0; // area
		unsigned TP = 0; // true positives
		unsigned FP = 0; // false positives
		unsigned TPPrev = 0; // previous true positives
		unsigned FPPrev = 0; // previous false positives
		double   predictionPrev = -std::numeric_limits<double>::max(); // previous prediction
		for(std::size_t i=0; i != elements; i++)  {
			if(L[i].key != predictionPrev){
				A += trapArea(FP,FPPrev,TP,TPPrev);
				predictionPrev = L[i].key;
				FPPrev = FP;
				TPPrev = TP;
			}
			if(L[i].value > 0) 
				TP++; // positive example
			else 
				FP++; // negative example
		}
		// deviation from the original algorithm description: A += trapArea(1, FPPrev, 1, TPPrev);
		A += trapArea(FP, FPPrev, TP, TPPrev);

		A /= double(N*P);
		return -A;
	}
	/// \brief Computes area under the curve. If the prediction vector is
	/// 1-dimensional, the "positive" class is mapped to larger values. If
	/// the prediction vector is 2-dimensional, the second dimension is
	/// viewed as the "positive" class. For higher dimensional vectors, an
	/// exception is thrown. In such a case, the column has to be
	/// explicitly specified as an additional parameter.
	///
	/// \param target: class label, 0 or 1
	/// \param prediction: prediction by classifier, OutputType-valued vector
	double eval(Data<LabelType> const& target, Data<OutputType>  const& prediction) const {
		SHARK_CHECK(prediction.numberOfElements() >= 1,"[NegativeAUC::eval] empty prediction set");
		
		std::size_t dim = dataDimension(prediction);
		if(dim == 1) 
			return eval(target, prediction, 0);
		else if(dim == 2) 
			return eval(target, prediction, 1);
		
		throw SHARKEXCEPTION("[NegativeAUC::eval] no default value for column");
		return 0.;
	}


 protected:
	double trapArea(double x1, double x2, double y1, double y2) const {
		double base = std::abs(x1-x2);
		double heightAvg = (y1+y2)/2;
		return base * heightAvg;
	}

	bool m_invert;
};

///
/// \brief Negative Wilcoxon-Mann-Whitney statistic 
/// 
/// This class computes the Wilcoxon-Mann-Whitney statistic, which is
/// an unbiased estimate of the area under the ROC curve.
///
/// See, for example:
/// Corinna Cortes, Mehryar Mohri. Confidence Intervals for the Area under the ROC Curve. NIPS, 2004
///
/// The area is negated so that optimizing the AUC corresponds to a minimization task. 
///
template<class LabelType = unsigned int, class OutputType = LabelType>
class NegativeWilcoxonMannWhitneyStatistic : public AbstractCost<LabelType, OutputType>
{
 public:
	/// Constructor.
	/// \param invert: if set to true, the role of positive and negative class are switched
	NegativeWilcoxonMannWhitneyStatistic(bool invert = false) {
		m_invert = invert;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NegativeWilcoxonMannWhitneyStatistic"; }

	/// \brief Computes Wilcoxon-Mann-Whitney statistic.
	/// \param target: interpreted as binary class label
	/// \param prediction: interpreted as binary class label
	/// \param column: indicates the column of the prediction vector interpreted as probability of positive class
	double eval(Data<LabelType> const& target, Data<OutputType> const& prediction, unsigned int column) const {
		SHARK_CHECK(prediction(0).size() > column,"[NegativeWilcoxonMannWhitneyStatistic::eval] column number too large");
		std::vector<double> pos, neg;
		for(std::size_t i=0; i<prediction.size(); i++) {
			if(!m_invert){
				if(target(i) > 0) 
					pos.push_back(prediction.element(i)(column));
				else  
					neg.push_back(prediction.element(i)(column));
			}else{
				if(target(i) > 0)
					pos.push_back(-prediction.element(i)(column));
				else
					neg.push_back(-prediction.element(i)(column));
			}
		}
		std::size_t m = pos.size();
		std::size_t n = neg.size();
		
		std::sort (pos.begin(), pos.end());
		std::sort (neg.begin(), neg.end());
		
		double A = 0;
		for(std::size_t i = 0, j = 0; i != m; i++) {
			A += j; 
			for(std::size_t j = 0; j != n; j++) {
				if(pos[i] > neg[j]) 
					A++;
				else 
					break;
			}
		}
		
#ifdef DEBUG
		// most naive implementation 
		double verifyA = 0.;
		for(std::size_t i=0; i<m; i++) {
			for(std::size_t  j=0; j<n; j++) {
				if(pos[i] > neg[j]) verifyA++;
			}
		}
		if (A!=verifyA) {
			throw( shark::Exception( "shark::WilcoxonMannWhitneyStatistic::eval: error in algorithm, efficient and naive implementation do no coincide", __FILE__, __LINE__ ) );
		}
#endif
		
		return -A / (n*m);
	}

	double eval(Data<LabelType> const& target, Data<OutputType>  const& prediction) const {
		SHARK_CHECK(prediction.numberOfElements() >= 1,"[NegativeAUC::eval] empty prediction set");
		
		std::size_t dim = dataDimension(prediction);
		if(dim == 1) 
			return eval(target, prediction, 0);
		else if(dim == 2) 
			return eval(target, prediction, 1);
		
		throw SHARKEXCEPTION("[NegativeAUC::eval] no default value for column");
		return 0.;
	}
private:
	bool m_invert;
};


}
#endif
