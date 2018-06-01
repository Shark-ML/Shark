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
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/Algorithms/GradientDescent/BFGS.h>

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
		ClassificationDataset validation_dataset;
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
			validation_dataset.append(transformInputs(m_folds.validation(i),svm.decisionFunction()));
		}

		// Fit a logistic regression to the prediction
		LinearModel<> logistic_model = fitLogistic(validation_dataset);
		
		//to evaluate, we use cross entropy loss on the fitted model 
		CrossEntropy<unsigned int, RealVector> logistic_loss;
		return logistic_loss(validation_dataset.labels(),logistic_model(validation_dataset.inputs()));
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
		// these two will be filled in order corresp. to all CV validation partitions stacked
		// behind one another, and then used to create datasets with
		std::vector< unsigned int > tmp_helper_labels(m_numSamples);
		std::vector< RealVector > tmp_helper_preds(m_numSamples);

		unsigned int next_label = 0; //helper index counter to monitor the next position to be filled in the above vectors
		// init variables especially for derivative
		RealMatrix all_validation_predict_derivs(m_numSamples, m_nhp);   //will hold derivatives of all output scores w.r.t. all hyperparameters
		RealVector der; //temporary helper for derivative calls

		// for each fold, train an svm and get predictions on the validation data
		for (std::size_t i=0; i<m_numFolds; i++) {
			// get current train/validation partitions as well as corresponding labels
			ClassificationDataset cur_train_data = m_folds.training(i);
			ClassificationDataset cur_valid_data = m_folds.validation(i);
			std::size_t cur_vsize = cur_valid_data.numberOfElements();
			Data< unsigned int > cur_vlabels = cur_valid_data.labels(); //validation labels of this fold
			Data< RealVector > cur_vinputs = cur_valid_data.inputs(); //validation inputs of this fold
			Data< RealVector > cur_vscores; //will hold SVM output scores for current validation partition
			// init SVM
			KernelClassifier<InputType> svm;   //the SVM
			CSvmTrainer<InputType, double> csvm_trainer(mep_kernel, C_reg, true, m_svmCIsUnconstrained);   //the trainer
			csvm_trainer.sparsify() = false;
			csvm_trainer.setComputeBinaryDerivative(true);
			if (mep_svmStoppingCondition != NULL) {
				csvm_trainer.stoppingCondition() = *mep_svmStoppingCondition;
			}
			// train SVM on current fold
			csvm_trainer.train(svm, cur_train_data);
			CSvmDerivative<InputType> svm_deriv(&svm, &csvm_trainer);
			cur_vscores = svm.decisionFunction()(cur_valid_data.inputs());   //will result in a dataset of RealVector as output
			// copy the scores and corresponding labels to the dataset-wide storage
			for (std::size_t j=0; j<cur_vsize; j++) {
				// copy label and prediction score
				tmp_helper_labels[next_label] = cur_vlabels.element(j);
				tmp_helper_preds[next_label] = cur_vscores.element(j);
				// get and store the derivative of the score w.r.t. the hyperparameters
				svm_deriv.modelCSvmParameterDerivative(cur_vinputs.element(j), der);
				noalias(row(all_validation_predict_derivs, next_label)) = der;   //fast assignment of the derivative to the correct matrix row
				++next_label;
			}
		}
		
		// now we got it all: the predictions across the validation folds, plus the correct corresponding
		// labels. so we go ahead and fit a logistic regression
		ClassificationDataset validation_dataset= createLabeledDataFromRange(tmp_helper_preds, tmp_helper_labels);
		LinearModel<> logistic_model = fitLogistic(validation_dataset);
		
		// to evaluate, we use cross entropy loss on the fitted model  and compute 
		// the derivative wrt the svm model parameters.
		derivative.resize(m_nhp);
		derivative.clear();
		double error = 0;
		std::size_t start = 0;
		for(auto const& batch: validation_dataset.batches()){
			std::size_t end = start+batch.size();
			CrossEntropy<unsigned int, RealVector> logistic_loss;
			RealMatrix lossGradient;
			error += logistic_loss.evalDerivative(batch.label,logistic_model(batch.input),lossGradient);
			noalias(derivative) += column(lossGradient,0) % rows(all_validation_predict_derivs,start,end);
			start = end;
		}
		derivative *= logistic_model.parameterVector()(0);
		derivative /= m_numSamples;
		return error / m_numSamples;
	}
private:
	LinearModel<> fitLogistic(ClassificationDataset const& data)const{
		LinearModel<> logistic_model;
		logistic_model.setStructure(1,1, true);//1 input, 1 output, bias = 2 parameters
		CrossEntropy<unsigned int, RealVector> logistic_loss;
		ErrorFunction<> error(data, &logistic_model, & logistic_loss);
		BFGS<> optimizer;
		optimizer.init(error);
		//this converges after very few iterations (typically 20 function evaluations)
		while(norm_2(optimizer.derivative())> 1.e-8){
			double lastValue = optimizer.solution().value;
			optimizer.step(error);
			if(lastValue == optimizer.solution().value) break;//we are done due to numerical precision
		}
		logistic_model.setParameterVector(optimizer.solution().point);
		return logistic_model;
	}
};


}
#endif
