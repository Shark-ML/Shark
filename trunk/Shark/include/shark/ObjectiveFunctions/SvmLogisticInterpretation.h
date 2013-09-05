/*!
 *  \file SvmLogisticInterpretation.h
 *
 *  \brief Maximum-likelihood model selection for binary support vector machines.
 *
 *  \author M.Tuma, T.Glasmachers
 *  \date 2009-2012
 *
 *  \par Copyright (c) 2009-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_ML_SVMLOGISTICINTERPRETATION_H
#define SHARK_ML_SVMLOGISTICINTERPRETATION_H

#include <shark/Data/CVDatasetTools.h>
#include <shark/Models/Kernels/CSvmDerivative.h>
#include <shark/Algorithms/Trainers/SigmoidFit.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/NegativeClassificationLogLikelihood.h>


namespace shark {

///
/// \brief Maximum-likelihood model selection score for binary support vector machines
///
/// \par
/// This class implements the maximum-likelihood based SVM model selection
/// procedure presented in the article "Glasmachers and C. Igel. Maximum
/// Likelihood Model Selection for 1-Norm Soft Margin SVMs with Multiple
/// Parameters. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2010."
/// At this point, only binary C-SVMs are supported.
/// \par
/// This class implements an #AbstactObjectiveFunction. In detail, it provides
/// a differentiable measure of how well a C-SVM with given hyperparameters fulfills
/// the maximum-likelihood score presented in the paper. This error measure can then
/// be optimized for externally via gradient-based optimizers. In other words, this
/// class provides a score, not an optimization method or a training algorithm. The
/// C-SVM parameters have to be optimized with regard to this measure
///
template<class InputType = RealVector>
class SvmLogisticInterpretation : public AbstractObjectiveFunction< VectorSpace<double>, double > {
public:
	typedef AbstractObjectiveFunction< VectorSpace<double>, double > base_type;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef VectorSpace<double>::PointType PType; //mtq: what's the difference between this and the one above?
	typedef CVFolds< LabeledData<InputType, unsigned int> > FoldsType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	//mtq: other options in the bestiarium are ResultType and SecondOrderDerivative in KTA.h; TrainerType in NGPE.h; ModelType and CostType from CVE.h ..

protected:
	FoldsType m_folds;          ///< the underlying partitioned dataset.
	KernelType *mep_kernel;     ///< the kernel with which to run the SVM
	unsigned int m_nhp;         ///< for convenience, the Number of Hyper Parameters
	unsigned int m_nkp;         ///< for convenience, the Number of Kernel Parameters
	unsigned int m_numFolds;    ///< the number of folds to be used in cross-validation
	unsigned int m_numSamples;  ///< overall number of samples in the dataset
	unsigned int m_inputDims;   ///< input dimensionality
	bool m_svmCIsUnconstrained; ///< the SVM regularization parameter C is passed for unconstrained optimization, and the derivative should compensate for that
	QpStoppingCondition *mep_svmStoppingCondition; ///< the stopping criterion that is to be passed to the SVM trainer.
	bool m_sigmoidSlopeIsUnconstrained; ///< whether or not to use the unconstrained variant of the sigmoid. currently always true, not user-settable, existing for safety.

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
	,  m_numSamples(folds.numberOfElements())
	,  m_inputDims(inputDimension(folds.dataset()))
	,  m_svmCIsUnconstrained(unconstrained)
	,  mep_svmStoppingCondition(stop_cond)
	,  m_sigmoidSlopeIsUnconstrained(true)
	{
		SHARK_CHECK(kernel != NULL, "[SvmLogisticInterpretation::SvmLogisticInterpretation] kernel is not allowed to be NULL");  //mtq: necessary despite indirect check via call in initialization list?
		SHARK_CHECK(m_numFolds > 1, "[SvmLogisticInterpretation::SvmLogisticInterpretation] please provide a meaningful number of folds for cross validation");
		if (!m_svmCIsUnconstrained)   //mtq: important: we additionally need to deal with kernel feasibility indicators! important!
			this->m_features|=base_type::IS_CONSTRAINED_FEATURE;
		this->m_features|=base_type::HAS_VALUE;
		if (mep_kernel->hasFirstParameterDerivative())
			this->m_features|=base_type::HAS_FIRST_DERIVATIVE;
		this->m_features|=base_type::CAN_PROPOSE_STARTING_POINT;
		this->m_features|=base_type::CAN_PROVIDE_CLOSEST_FEASIBLE;
		m_folds = folds;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SvmLogisticInterpretation"; }

	//! checks whether the search point provided is feasible
	//! \param input the point to test for feasibility
	bool isFeasible(const SearchPointType &input) const {
		SHARK_ASSERT(input.size() == m_nhp);
		//throw SHARKEXCEPTION("[SvmLogisticInterpretation::isFeasible] Please first clarify how the kernel parameter feasibility should be dealt with. Afterwards, please write a test for this method. Thanks.");
		if (input(0) <= 0.0 && !m_svmCIsUnconstrained) {
			return false;
		}
		return true;
	}

	//! propose a starting point to an external optimizer. we use the current kernel parameters and a regularization parameter of 1.
	//! \param startingPoint the proposed point will be stored in this variable
	void proposeStartingPoint(SearchPointType &startingPoint) const {
		startingPoint.resize(m_nhp);
		startingPoint(m_nkp) = 1.0;   //set C to arbitrary value of 1, regardless of constrained-encoding-ness
		SearchPointType tmp_params = mep_kernel->parameterVector();
		for (unsigned int k=0; k<m_nkp; k++) {   //set kernel parameters to current kernel parameters (assuming there are some)
			startingPoint(k) = tmp_params(k);
		}
		throw SHARKEXCEPTION("[SvmLogisticInterpretation::proposeStartingPoint] Please first clarify how the kernel parameter feasibility should be dealt with. Afterwards, please write a test for this method. Thanks.");
	}

	//! repair a non-feasible point so that it becomes feasible
	//! \param input the non-feasible point to repair
	void closestFeasible(SearchPointType &input) const {
		SHARK_ASSERT(input.size() == m_nhp);
		if (input(m_nkp) <= 0.0 && !m_svmCIsUnconstrained) {
			input(m_nkp) = 1e-10;   //should be an alright value to set C to..
		} //else: leave everything as is
		throw SHARKEXCEPTION("[SvmLogisticInterpretation::closestFeasible] Please first clarify how the kernel parameter feasibility should be dealt with. Afterwards, please write a test for this method. Thanks.");
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
		SHARK_CHECK(m_nhp == parameters.size(), "[SvmLogisticInterpretation::eval] wrong number of parameters");
		// initialize, copy parameters
		double C_reg = (m_svmCIsUnconstrained ? std::exp(parameters(m_nkp)) : parameters(m_nkp));   //set up regularization parameter
		mep_kernel->setParameterVector(subrange(parameters, 0, m_nkp));   //set up kernel parameters
		// these two will be filled in order corresp. to all CV validation partitions stacked
		// behind one another, and then used to create datasets with
		std::vector< unsigned int > tmp_helper_labels(m_numSamples);
		std::vector< RealVector > tmp_helper_preds(m_numSamples);
		unsigned int next_label = 0; //helper index counter to monitor the next position to be filled in the above vectors

		// for each fold, train an svm and get predictions on the validation data
		for (unsigned int i=0; i<m_numFolds; i++) {
			// get current train/validation partitions as well as corresponding labels
			ClassificationDataset cur_train_data = m_folds.training(i);
			ClassificationDataset cur_valid_data = m_folds.validation(i);
			unsigned int cur_vsize = cur_valid_data.numberOfElements();
			Data< unsigned int > cur_vlabels = cur_valid_data.labels(); //validation labels of this fold
			Data< RealVector > cur_vscores; //will hold SVM output scores for current validation partition
			// init SVM
			KernelExpansion<InputType> svm(mep_kernel);   //the SVM
			CSvmTrainer<InputType, double> csvm_trainer(mep_kernel, C_reg, m_svmCIsUnconstrained);   //the trainer
			csvm_trainer.sparsify() = false;
			if (mep_svmStoppingCondition != NULL) {
				csvm_trainer.stoppingCondition() = *mep_svmStoppingCondition;
			} else {
				csvm_trainer.stoppingCondition().minAccuracy = 1e-3; //mtq: is this necessary? i think it could be set via long chain of default ctors..
				csvm_trainer.stoppingCondition().maxIterations = 200 * m_inputDims; //mtq: need good/better heuristics to determine a good value for this
			}

			// train SVM on current fold
			csvm_trainer.train(svm, cur_train_data);
			cur_vscores = svm(cur_valid_data.inputs());   //will result in a dataset of RealVector as output
			// copy the scores and corresponding labels to the dataset-wide storage
			for (unsigned int j=0; j<cur_vsize; j++) {
				tmp_helper_labels[next_label] = cur_vlabels.element(j);
				tmp_helper_preds[next_label] = cur_vscores.element(j);
				++next_label;
			}
		}
		Data< unsigned int > all_validation_labels = createDataFromRange(tmp_helper_labels);
		Data< RealVector > all_validation_predictions = createDataFromRange(tmp_helper_preds);

		// now we got it all: the predictions across the validation folds, plus the correct corresponding
		// labels. so we go ahead and fit a sigmoid to be as good as possible a model between the two:
		SigmoidModel sigmoid_model(m_sigmoidSlopeIsUnconstrained);   //use the unconstrained variant?
		SigmoidFitRpropNLL sigmoid_trainer(100);   //number of rprop iterations
		ClassificationDataset validation_dataset(all_validation_predictions, all_validation_labels);
		sigmoid_trainer.train(sigmoid_model, validation_dataset);
		// we're basically done. now only get the final cost value of the best fit, and return it:
		Data< RealVector > sigmoid_predictions = sigmoid_model(all_validation_predictions);
		NegativeClassificationLogLikelihood ncll;
		return ncll.eval(all_validation_labels, sigmoid_predictions);
	}

	//! the derivative of the error() function above w.r.t. the parameters.
	//! \param parameters the SVM hyperparameters to use for all C-SVMs
	//! \param derivative will store the computed derivative w.r.t. the current hyperparameters
	// mtq: should this also follow the first-call-error()-then-call-deriv() paradigm?
	double evalDerivative(PType const &parameters, FirstOrderDerivative &derivative) const {
		SHARK_CHECK(m_nhp == parameters.size(), "[SvmLogisticInterpretation::evalDerivative] wrong number of parameters");
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
		for (unsigned int i=0; i<m_numFolds; i++) {
			// get current train/validation partitions as well as corresponding labels
			ClassificationDataset cur_train_data = m_folds.training(i);
			ClassificationDataset cur_valid_data = m_folds.validation(i);
			unsigned int cur_vsize = cur_valid_data.numberOfElements();
			Data< unsigned int > cur_vlabels = cur_valid_data.labels(); //validation labels of this fold
			Data< RealVector > cur_vinputs = cur_valid_data.inputs(); //validation inputs of this fold
			Data< RealVector > cur_vscores; //will hold SVM output scores for current validation partition
			// init SVM
			KernelExpansion<InputType> svm(mep_kernel);   //the SVM
			CSvmTrainer<InputType, double> csvm_trainer(mep_kernel, C_reg, m_svmCIsUnconstrained);   //the trainer
			csvm_trainer.sparsify() = false;
			if (mep_svmStoppingCondition != NULL) {
				csvm_trainer.stoppingCondition() = *mep_svmStoppingCondition;
			} else {
				csvm_trainer.stoppingCondition().maxIterations = 200 * m_inputDims; //mtq: need good/better heuristics to determine a good value for this
			}
			// train SVM on current fold
			csvm_trainer.train(svm, cur_train_data);
			CSvmDerivative<InputType> svm_deriv(&svm, &csvm_trainer);
			cur_vscores = svm(cur_valid_data.inputs());   //will result in a dataset of RealVector as output
			// copy the scores and corresponding labels to the dataset-wide storage
			for (unsigned int j=0; j<cur_vsize; j++) {
				// copy label and prediction score
				tmp_helper_labels[next_label] = cur_vlabels.element(j);
				tmp_helper_preds[next_label] = cur_vscores.element(j);
				// get and store the derivative of the score w.r.t. the hyperparameters
				svm_deriv.modelCSvmParameterDerivative(cur_vinputs.element(j), der);
				noalias(row(all_validation_predict_derivs, next_label)) = der;   //fast assignment of the derivative to the correct matrix row
				++next_label;
			}
		}
		Data< unsigned int > all_validation_labels = createDataFromRange(tmp_helper_labels);
		Data< RealVector > all_validation_predictions = createDataFromRange(tmp_helper_preds);

		// now we got it all: the predictions across the validation folds, plus the correct corresponding
		// labels. so we go ahead and fit a sigmoid to be as good as possible a model between the two:
		SigmoidModel sigmoid_model(m_sigmoidSlopeIsUnconstrained);   //use the unconstrained variant?
		SigmoidFitRpropNLL sigmoid_trainer(100);   //number of rprop iterations
		ClassificationDataset validation_dataset(all_validation_predictions, all_validation_labels);
		sigmoid_trainer.train(sigmoid_model, validation_dataset);
		// we're basically done. now only get the final cost value of the best fit, and return it:
		Data< RealVector > sigmoid_predictions = sigmoid_model(all_validation_predictions);
		NegativeClassificationLogLikelihood ncll;

		// finally compute the derivative of the sigmoid model predictions:
		// (we're here a bit un-shark-ish in that we do some of the derivative calculations by hand where they
		//  would also and more consistently be offered by their respective classes. one reason we're doing it
		//  like this might be the missing batch processing for the evalDerivatives)
		derivative.resize(m_nhp);
		zero(derivative);

		double ss = (m_sigmoidSlopeIsUnconstrained ? std::exp(sigmoid_model.parameterVector()(0)) : sigmoid_model.parameterVector()(0));
		for (unsigned int i=0; i<m_numSamples; i++) {
			double p = sigmoid_predictions.element(i)(0);
			// compute derivative of the negative log likelihood
			double dL_dsp; //derivative of likelihood wrt sigmoid predictions
			if (all_validation_labels.element(i) == 1)   //positive class
				dL_dsp = -1.0/p;
			else //negative class
				dL_dsp = 1.0/(1.0-p);
			// compute derivative of the sigmoid
			// derivative of sigmoid predictions wrt svm predictions
			double dsp_dsvmp = ss * p * (1.0-p); //severe sign confusion potential: p(1-p) is deriv. w.r.t. t in 1/(1+e**(-t))!
			for (unsigned int j=0; j<m_nhp; j++) {
				derivative(j) += dL_dsp * dsp_dsvmp * all_validation_predict_derivs(i,j);
			}
		}
		// correct for the fact that AbstractLoss devides the NCLL eval scores 
		// by the number of examples when evaluated in batch mode
		derivative /= m_numSamples;
		return ncll.eval(all_validation_labels, sigmoid_predictions);
	}
};


}
#endif
