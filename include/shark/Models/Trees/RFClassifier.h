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
#include <shark/Models/Classifier.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>        
#include <shark/Data/DataView.h>

namespace shark {

namespace detail{
//this class bridges the differences between random forests in classification and regression
template<class LabelType>
class RFClassifierBase : public MeanModel<CARTree<LabelType> >{
protected:
	double doComputeOOBerror(
		UIntMatrix const& oobPoints, LabeledData<RealVector, RealVector> const& data
	){
		double OOBerror = 0;
		//aquire votes for every element
		RealVector mean(labelDimension(data));
		RealVector input(inputDimension(data));
		std::size_t elem = 0;
		for(auto const& point: data.elements()){
			noalias(input) = point.input;
			mean.clear();
			std::size_t oobModels = 0;
			for(std::size_t m = 0; m != this->numberOfModels();++m){
				if(oobPoints(m,elem)){
					++oobModels;
					auto const& model = this->getModel(m);
					noalias(mean) += model(input);
				}
			}
			mean /= oobModels;
			OOBerror += 0.5 * norm_sqr(point.label - mean);
			++elem;
		}
		OOBerror /= data.numberOfElements();
		return OOBerror;
	}
	
	double loss(RealMatrix const& labels, RealMatrix const& predictions) const{
		SquaredLoss<RealVector, RealVector> loss;
		return loss.eval(labels,  predictions);
	}
};

template<>
class RFClassifierBase<unsigned int> : public Classifier<MeanModel<CARTree<unsigned int> > >{
public:
	//make the interface of MeanModel publicly available for same basic interface for classification and regression case
	CARTree<unsigned int> const& getModel(std::size_t index)const{
		return this->decisionFunction().getModel(index);
	}
	
	void addModel(CARTree<unsigned int> const& model, double weight = 1.0){
		this->decisionFunction().addModel(model,weight);
	}
	void clearModels(){
		this->decisionFunction().clearModels();
	}
	
	void setOutputSize(std::size_t dim){
		this->decisionFunction().setOutputSize(dim);
	}
	
	/// \brief Returns the number of models.
	std::size_t numberOfModels()const{
		return this->decisionFunction().numberOfModels();
	}
protected:
	double loss(UIntVector const& labels, UIntVector const& predictions) const{
		ZeroOneLoss<unsigned int> loss;
		return loss.eval(labels,  predictions);
	}
	
	double doComputeOOBerror(
		UIntMatrix const& oobPoints, LabeledData<RealVector, unsigned int> const& data
	){
		double OOBerror = 0;
		//aquire votes for every element
		RealVector votes(numberOfClasses(data));
		RealVector input(inputDimension(data));
		std::size_t elem = 0;
		for(auto const& point: data.elements()){
			noalias(input) = point.input;
			votes.clear();
			for(std::size_t m = 0; m != numberOfModels();++m){
				if(oobPoints(m,elem)){
					auto const& model = getModel(m);
					unsigned int label = model(input);
					votes(label) += 1;
				}
			}
			OOBerror += (arg_max(votes) != point.label);
			++elem;
		}
		OOBerror /= data.numberOfElements();
		return OOBerror;
	}
};
}

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
class RFClassifier : public detail::RFClassifierBase<LabelType>
{
public:
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
		std::size_t n = this->numberOfModels();
		if(!n) return UIntVector();
		UIntVector r = this->getModel(0).countAttributes();
		for(std::size_t i=1; i< n; i++ ) {
			noalias(r) += this->getModel(i).countAttributes();
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
		m_OOBerror = this->doComputeOOBerror(oobMatrix,data);
	}

	/// Compute feature importances, given an oob dataset
	///
	/// For each tree, extracts the out-of-bag-samples indicated by oobIndices. The feature importance is defined
	/// as the average change of loss (Squared loss or accuracy depending on label type) when the feature is permuted across the oob samples of a tree.
	void computeFeatureImportances(std::vector<std::vector<std::size_t> > const& oobIndices, LabeledData<RealVector, LabelType> const& data, random::rng_type& rng){
		std::size_t inputs = inputDimension(data);
		m_featureImportances.resize(inputs);
		DataView<LabeledData<RealVector, LabelType> const > view(data);
		
		for(std::size_t m = 0; m != this->numberOfModels();++m){
			auto batch = subBatch(view, oobIndices[m]);
			double errorBefore = this->loss(batch.label,this->getModel(m)(batch.input));
			
			for(std::size_t i=0; i!=inputs;++i) {
				RealVector vOld= column(batch.input,i);
				RealVector v = vOld;
				std::shuffle(v.begin(), v.end(), rng);
				noalias(column(batch.input,i)) = v;
				double errorAfter = this->loss(batch.label,this->getModel(m)(batch.input));
				noalias(column(batch.input,i)) = vOld;
				m_featureImportances(i) += (errorAfter - errorBefore) / batch.size();
			}
		}
		m_featureImportances /= this->numberOfModels();
	}

private:
	double m_OOBerror; ///< oob error for the forest
	RealVector m_featureImportances; ///< feature importances for the forest

};


}
#endif
