/*!
 * 
 *
 * \brief       Maximum-likelihood model selection for binary support vector machines.
 * 
 * 
 *
 * \author      M.Tuma, T.Glasmachers
 * \date        2009-2012
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
#ifndef SHARK_ML_SVMLOGISTICINTERPRETATION_H
#define SHARK_ML_SVMLOGISTICINTERPRETATION_H

#include <shark/Data/CVDatasetTools.h>
#include <shark/Models/Kernels/CSvmDerivative.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/Trainers/LogisticRegression.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>

namespace shark {

/// \brief Maximum-likelihood model selection score for binary support vector machines
///
/// \par
/// This class implements the maximum-likelihood based SVM model selection
/// procedure presented in the article "Glasmachers and C. Igel. Maximum
/// Likelihood Model Selection for 1-Norm Soft Margin SVMs with Multiple
/// Parameters. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2010."
/// At this point, only binary C-SVMs are supported.
/// \par
/// This class implements an AbstactObjectiveFunction. In detail, it provides
/// a differentiable measure of how well a C-SVM with given hyperparameters fulfills
/// the maximum-likelihood score presented in the paper. This error measure can then
/// be optimized for externally via gradient-based optimizers. In other words, this
/// class provides a score, not an optimization method or a training algorithm. The
/// C-SVM parameters have to be optimized with regard to this measure
/// \ingroup kerneloptimization
template<class InputType = RealVector>
class SvmLogisticInterpretation : public AbstractObjectiveFunction< RealVector, double > {
public:
	typedef CVFolds< LabeledData<InputType, unsigned int> > FoldsType;
	typedef AbstractKernelFunction<InputType> KernelType;
protected:
	FoldsType m_folds;          ///< the underlying partitioned dataset.
	KernelType *mep_kernel;     ///< the kernel with which to run the SVM
	std::size_t m_nhp;         ///< for convenience, the Number of Hyper Parameters
	std::size_t m_nkp;         ///< for convenience, the Number of Kernel Parameters
	std::size_t m_numFolds;    ///< the number of folds to be used in cross-validation
	std::size_t m_numSamples;  ///< overall number of samples in the dataset
	std::size_t m_inputDims;   ///< input dimensionality
	bool m_svmCIsUnconstrained; ///< the SVM regularization parameter C is passed for unconstrained optimization, and the derivative should compensate for that
	QpStoppingCondition *mep_svmStoppingCondition; ///< the stopping criterion that is to be passed to the SVM trainer.
public:

	//! constructor.
	//! \param folds an already partitioned dataset (i.e., a CVFolds object)
	//! \param kernel pointer to the kernel to be used within the SVMs.
	//! \param unconstrained whether or not the C-parameter of/for the C-SVM is passed for unconstrained optimization mode.
	//! \param stop_cond the stopping conditions which are to be passed to the
	SvmLogisticInterpretation(
		FoldsType const &folds, KernelType *kernel,
	        bool unconstrained = true, QpStoppingCondition *stop_cond = NULL
	)
	: mep_kernel(kernel)
	,  m_nhp(kernel->parameterVector().size()+1)
	,  m_nkp(kernel->parameterVector().size())
	,  m_numFolds(folds.size())  //gets number of folds!
	,  m_numSamples(folds.dataset().numberOfElements())
	,  m_inputDims(inputDimension(folds.dataset()))
	,  m_svmCIsUnconstrained(unconstrained)
	,  mep_svmStoppingCondition(stop_cond)
	{
		SHARK_RUNTIME_CHECK(kernel != NULL, "[SvmLogisticInterpretation::SvmLogisticInterpretation] kernel is not allowed to be NULL");  //mtq: necessary despite indirect check via call in initialization list?
		SHARK_RUNTIME_CHECK(m_numFolds > 1, "[SvmLogisticInterpretation::SvmLogisticInterpretation] please provide a meaningful number of folds for cross validation");
		if (!m_svmCIsUnconstrained)   //mtq: important: we additionally need to deal with kernel feasibility indicators! important!
			m_features|=IS_CONSTRAINED_FEATURE;
		m_features|=HAS_VALUE;
		if (mep_kernel->hasFirstParameterDerivative())
			m_features|=HAS_FIRST_DERIVATIVE;
		m_folds = folds;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SvmLogisticInterpretation"; }

	//! checks whether the search point provided is feasible
	//! \param input the point to test for feasibility
	bool isFeasible(const SearchPointType &input) const {
		SHARK_ASSERT(input.size() == m_nhp);
		if (input(0) <= 0.0 && !m_svmCIsUnconstrained) {
			return false;
		}
		return true;
	}

	std::size_t numberOfVariables()const{
		return m_nhp;
	}

	//! train a number of SVMs in a cross-validation setting using the hyperparameters passed to this method.
	//! the output scores from all validations sets are then concatenated. together with the true labels, these
	//! scores can then be used to fit a sigmoid such that it becomes as good as possible a model for the
	//! class membership probabilities given the SVM output scores. This method returns the negative likelihood
	//! of the best fitting sigmoid, given a set of SVM hyperparameters.
	//! \param parameters the SVM hyperparameters to use for all C-SVMs
	double eval(SearchPointType const &parameters) const {
		SHARK_RUNTIME_CHECK(m_nhp == parameters.size(), "[SvmLogisticInterpretation::eval] wrong number of parameters");
		// initialize, copy parameters
		double C_reg = (m_svmCIsUnconstrained ? std::exp(parameters(m_nkp)) : parameters(m_nkp));   //set up regularization parameter
		mep_kernel->setParameterVector(subrange(parameters, 0, m_nkp));   //set up kernel parameters
		// Stores the stacked CV predictions for every fold.
		ClassificationDataset validation_scores;
		// for each fold, train an svm and get predictions on the validation data
		for (std::size_t i=0; i<m_numFolds; i++) {
			// init SVM
			KernelClassifier<InputType> svm;
			CSvmTrainer<InputType, double> csvm_trainer(mep_kernel, C_reg, true, m_svmCIsUnconstrained);   //the trainer
			csvm_trainer.sparsify() = false;
			if (mep_svmStoppingCondition != NULL) {
				csvm_trainer.stoppingCondition() = *mep_svmStoppingCondition;
			}

			// train SVM on current training fold
			csvm_trainer.train(svm, m_folds.training(i));
			
			//append validation predictions
			validation_scores.append(transformInputs(m_folds.validation(i),svm.decisionFunction()));
		}

		// Fit a logistic regression to the prediction
		LogisticRegression<RealVector> logistic_trainer(0.0,0.0,true);
		LinearClassifier<RealVector> logistic_model;
		logistic_trainer.train(logistic_model, validation_scores);
		
		//to evaluate, we use cross entropy loss on the fitted model 
		CrossEntropy<unsigned int, RealVector> logistic_loss;
		return logistic_loss(validation_scores.labels(),logistic_model.decisionFunction()(validation_scores.inputs()));
	}

	//! the derivative of the error() function above w.r.t. the parameters.
	//! \param parameters the SVM hyperparameters to use for all C-SVMs
	//! \param derivative will store the computed derivative w.r.t. the current hyperparameters
	// mtq: should this also follow the first-call-error()-then-call-deriv() paradigm?
	double evalDerivative(SearchPointType const &parameters, FirstOrderDerivative &derivative) const {
		SHARK_RUNTIME_CHECK(m_nhp == parameters.size(), "[SvmLogisticInterpretation::evalDerivative] wrong number of parameters");
		// initialize, copy parameters
		double C_reg = (m_svmCIsUnconstrained ? std::exp(parameters(m_nkp)) : parameters(m_nkp));   //set up regularization parameter
		mep_kernel->setParameterVector(subrange(parameters, 0, m_nkp));   //set up kernel parameters
		
		//holds the pairs of scores and real labels for all validation folds
		//this is going to be the dataset over which a logistic regression is fitted
		ClassificationDataset validation_scores;

		unsigned int next_label = 0; //helper index counter to monitor the next position to be filled in the above vectors
		// init variables especially for derivative
		RealMatrix all_validation_predict_derivs(m_numSamples, m_nhp);   //will hold derivatives of all output scores w.r.t. all hyperparameters
		
		// for each fold, train an svm and get predictions on the validation data
		for (std::size_t i=0; i<m_numFolds; i++) {
			// train svm using the training part of the folg
			KernelClassifier<InputType> svm;   //the SVM
			CSvmTrainer<InputType, double> csvm_trainer(mep_kernel, C_reg, true, m_svmCIsUnconstrained);   //the trainer
			csvm_trainer.sparsify() = false;
			csvm_trainer.setComputeBinaryDerivative(true);
			if (mep_svmStoppingCondition != NULL) {
				csvm_trainer.stoppingCondition() = *mep_svmStoppingCondition;
			}
			csvm_trainer.train(svm, m_folds.training(i));
			
			// copy the predictions and corresponding labels to the dataset-wide storage
			validation_scores.append(transformInputs(m_folds.validation(i),svm.decisionFunction()));
			
			//compute the derivative for each element in the validation dataset
			CSvmDerivative<InputType> svm_deriv(&svm, &csvm_trainer);
			RealVector der; //temporary helper for derivative calls
			LabeledData<InputType, unsigned int> validation = m_folds.validation(i);
			for (auto const& element: validation.elements()){
				// get and store the derivative of the score w.r.t. the hyperparameters
				svm_deriv.modelCSvmParameterDerivative(element.input, der);
				noalias(row(all_validation_predict_derivs, next_label)) = der;   //fast assignment of the derivative to the correct matrix row
				++next_label;
			}
		}
		
		// now we got it all: the predictions across the validation folds, plus the correct corresponding
		// labels. so we go ahead and fit a logistic regression
		LogisticRegression<RealVector> logistic_trainer(0.0,0.0,true);
		LinearClassifier<RealVector> logistic_model;
		logistic_trainer.train(logistic_model, validation_scores);
		
		// to evaluate, we use cross entropy loss on the fitted model  and compute 
		// the derivative wrt the svm model parameters.
		derivative.resize(m_nhp);
		derivative.clear();
		double error = 0;
		std::size_t start = 0;
		for(auto const& batch: validation_scores.batches()){
			std::size_t end = start+batch.size();
			CrossEntropy<unsigned int, RealVector> logistic_loss;
			RealMatrix lossGradient;
			error += logistic_loss.evalDerivative(batch.label,logistic_model.decisionFunction()(batch.input),lossGradient);
			noalias(derivative) += column(lossGradient,0) % rows(all_validation_predict_derivs,start,end);
			start = end;
		}
		derivative *= logistic_model.parameterVector()(0);
		derivative /= m_numSamples;
		return error / m_numSamples;
	}
};


}
#endif
