//===========================================================================
/*!
 * 
 *
 * \brief       Random Forest Classifier.
 * 
 * 
 *
 * \author      K. N. Hansen, O.Krause, J. Kremer
 * \date        2011-2012
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

#ifndef SHARK_MODELS_TREES_RFCLASSIFIER_H
#define SHARK_MODELS_TREES_RFCLASSIFIER_H

#include <shark/Models/Trees/CARTree.h>
#include <shark/Models/MeanModel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>        
#include <shark/Data/DataView.h>

namespace shark {

typedef CARTree<RealVector>::TreeType TreeType;
typedef std::vector<TreeType> ForestInfo;

///
/// \brief Random Forest Classifier.
///
/// \par
/// The Random Forest Classifier predicts a class label
/// using the Random Forest algorithm as described in<br/>
/// Random Forests. Leo Breiman. Machine Learning, 1(45), pages 5-32. Springer, 2001.<br/>
///
/// \par
/// It is an ensemble learner that uses multiple decision trees built
/// using the CART methodology.
///
template<class LabelType>
class RFClassifier : public MeanModel<CARTree<LabelType> >
{
public:
	using MeanModel<CARTree<LabelType> >::numberOfModels;
	using MeanModel<CARTree<LabelType> >::getModel;
	typedef CARTree<LabelType> SubmodelType;
	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RFClassifier"; }
	
	
	/// \brief Returns the computed out-of-bag-error of the forest
	double OOBerror() const {
		return m_OOBerror;
	}

	/// \brief Returns the computed feature importances of the forest
	RealVector const& featureImportances()const{
		return m_featureImportances;
	}

	/// \brief Counts how often attributes are used
	UIntVector countAttributes() const {
		std::size_t n = numberOfModels();
		if(!n) return UIntVector();
		UIntVector r = getModel(0).countAttributes();
		for(std::size_t i=1; i< n; i++ ) {
			noalias(r) += getModel(i).countAttributes();
		}
		return r;
	}


	
	
	/// Compute oob error, given an oob dataset (Classification)
	void computeOOBerror(std::vector<std::vector<std::size_t> > const& oobIndices, LabeledData<RealVector, LabelType> const& data){
		UIntMatrix oobMatrix(oobIndices.size(), data.numberOfElements(),0);
		for(std::size_t i = 0; i != oobMatrix.size1(); ++i){
			for(auto index: oobIndices[i])
				oobMatrix(i,index) = 1;
		}
		doComputeOOBerror(oobMatrix,data);
	}

	/// Compute feature importances, given an oob dataset
	///
	/// For each tree, extracts the out-of-bag-samples indicated by oobIndices. The feature importance is defined
	/// as the average change of loss (Squared loss or accuracy depending on label type) when the feature is permuted across the oob samples of a tree.
	void computeFeatureImportances(std::vector<std::vector<std::size_t> > const& oobIndices, LabeledData<RealVector, LabelType> const& data, random::rng_type& rng){
		std::size_t inputs = inputDimension(data);
		m_featureImportances.resize(inputs);
		DataView<LabeledData<RealVector, LabelType> const > view(data);
		
		for(std::size_t m = 0; m != numberOfModels();++m){
			auto batch = subBatch(view, oobIndices[m]);
			double errorBefore = loss(batch.label,getModel(m)(batch.input));
			
			for(std::size_t i=0; i!=inputs;++i) {
				RealVector vOld= column(batch.input,i);
				RealVector v = vOld;
				std::shuffle(v.begin(), v.end(), rng);
				noalias(column(batch.input,i)) = v;
				double errorAfter = loss(batch.label,getModel(m)(batch.input));
				noalias(column(batch.input,i)) = vOld;
				m_featureImportances(i) += (errorAfter - errorBefore) / batch.size();
			}
		}
		m_featureImportances /= numberOfModels();
	}

private:
	double loss(UIntVector const& labels, UIntVector const& predictions) const{
		ZeroOneLoss<unsigned int> loss;
		return loss.eval(labels,  predictions);
	}
	double loss(RealMatrix const& labels, RealMatrix const& predictions) const{
		SquaredLoss<RealVector, RealVector> loss;
		return loss.eval(labels,  predictions);
	}
	
	//different versions for different labels
	template<class Indices>
	void doComputeOOBerror(
		Indices const& oobPoints, LabeledData<RealVector, unsigned int> const& data
	){
		m_OOBerror = 0;
		//aquire votes for every element
		RealVector votes(numberOfClasses(data));
		RealVector input(inputDimension(data));
		std::size_t elem = 0;
		for(auto&& point: data.elements()){
			noalias(input) = point.input;
			votes.clear();
			for(std::size_t m = 0; m != numberOfModels();++m){
				if(oobPoints(m,elem)){
					auto const& model = getModel(m);
					unsigned int label = model(input);
					votes(label) += 1;
				}
			}
			m_OOBerror += (arg_max(votes) != point.label);
			++elem;
		}
		m_OOBerror /= data.numberOfElements();
	}
	
	template<class Indices>
	void doComputeOOBerror(
		Indices const& oobPoints, LabeledData<RealVector, RealVector> const& data
	){
		m_OOBerror = 0;
		//aquire votes for every element
		RealVector mean(labelDimension(data));
		RealVector input(inputDimension(data));
		std::size_t elem = 0;
		for(auto&& point: data.elements()){
			noalias(input) = point.input;
			mean.clear();
			std::size_t oobModels = 0;
			for(std::size_t m = 0; m != numberOfModels();++m){
				if(oobPoints(m,elem)){
					++oobModels;
					auto const& model = getModel(m);
					noalias(mean) += model(input);
				}
			}
			mean /= oobModels;
			m_OOBerror += 0.5 * norm_sqr(point.label - mean);
			++elem;
		}
		m_OOBerror /= data.numberOfElements();
	}
	

	
	double m_OOBerror; ///< oob error for the forest
	RealVector m_featureImportances; ///< feature importances for the forest

};


}
#endif
