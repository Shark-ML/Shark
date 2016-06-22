#ifndef SHARK_ALGORITHMS_CSVMTRAINER_H
#define SHARK_ALGORITHMS_CSVMTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/Trainers/AbstractWeightedTrainer.h>
#include <shark/Algorithms/QP/BoxConstrainedProblems.h>
#include <shark/Algorithms/QP/SvmProblems.h>
#include <shark/Algorithms/QP/QpBoxLinear.h>
#include <shark/LinAlg/CachedMatrix.h>
#include <shark/LinAlg/GaussianKernelMatrix.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PrecomputedMatrix.h>
#include <shark/LinAlg/RegularizedKernelMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

//all MCSVMs!
#include <shark/Algorithms/Trainers/McSvm/McSvmADMTrainer.h>
#include <shark/Algorithms/Trainers/McSvm/McSvmATMTrainer.h>
#include <shark/Algorithms/Trainers/McSvm/McSvmATSTrainer.h>
#include <shark/Algorithms/Trainers/McSvm/McSvmCSTrainer.h>
#include <shark/Algorithms/Trainers/McSvm/McSvmLLWTrainer.h>
#include <shark/Algorithms/Trainers/McSvm/McSvmWWTrainer.h>
#include <shark/Algorithms/Trainers/McSvm/McSvmMMRTrainer.h>
#include <shark/Algorithms/Trainers/McSvm/McReinforcedSvmTrainer.h>

namespace shark {
	
	
enum class McSvm{
	WW,
	CS,
	LLW,
	ATM,
	ATS,
	ADM,
	OVA,
	MMR,
	ReinforcedSvm
};


///
/// \brief Training of C-SVMs for binary classification.
///
/// The C-SVM is the "standard" support vector machine for
/// binary (two-class) classification. Given are data tuples
/// \f$ (x_i, y_i) \f$ with x-component denoting input and
/// y-component denoting the label +1 or -1 (see the tutorial on
/// label conventions; the implementation uses values 0/1),
/// a kernel function k(x, x') and a regularization
/// constant C > 0. Let H denote the kernel induced
/// reproducing kernel Hilbert space of k, and let \f$ \phi \f$
/// denote the corresponding feature map.
/// Then the SVM classifier is the function
/// \f[
///     h(x) = \mathop{sign} (f(x))
/// \f]
/// \f[
///     f(x) = \langle w, \phi(x) \rangle + b
/// \f]
/// with coefficients w and b given by the (primal)
/// optimization problem
/// \f[
///     \min \frac{1}{2} \|w\|^2 + C \sum_i L(y_i, f(x_i)),
/// \f]
/// where \f$ L(y, f(x)) = \max\{0, 1 - y \cdot f(x)\} \f$
/// denotes the hinge loss.
///
/// For details refer to the paper:<br/>
/// <p>Support-Vector Networks. Corinna Cortes and Vladimir Vapnik,
/// Machine Learning, vol. 20 (1995), pp. 273-297.</p>
/// or simply to the Wikipedia article:<br/>
/// http://en.wikipedia.org/wiki/Support_vector_machine
///
template <class InputType, class CacheType = float>
class CSvmTrainer : public AbstractSvmTrainer<
	InputType, unsigned int, 
	KernelClassifier<InputType>,
	AbstractWeightedTrainer<KernelClassifier<InputType> > 
>
{
private:
	typedef AbstractSvmTrainer<
		InputType, unsigned int, 
		KernelClassifier<InputType>,
		AbstractWeightedTrainer<KernelClassifier<InputType> > 
	> base_type;
public:

	/// \brief Convenience typedefs:
	/// this and many of the below typedefs build on the class template type CacheType.
	/// Simply changing that one template parameter CacheType thus allows to flexibly
	/// switch between using float or double as type for caching the kernel values.
	/// The default is float, offering sufficient accuracy in the vast majority
	/// of cases, at a memory cost of only four bytes. However, the template
	/// parameter makes it easy to use double instead, (e.g., in case high
	/// accuracy training is needed).
	typedef CacheType QpFloatType;

	typedef AbstractKernelFunction<InputType> KernelType;

	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param offset whether to train the svm with offset term
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	CSvmTrainer(KernelType* kernel, double C, bool offset, bool unconstrained = false)
	: base_type(kernel, C, offset, unconstrained), m_computeDerivative(false), m_McSvmType(McSvm::WW) //make  Vapnik happy!
	{ }
	
	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  negativeC   regularization parameter of the negative class (label 0)
	//! \param  positiveC    regularization parameter of the positive class (label 1)
	//! \param offset whether to train the svm with offset term
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	CSvmTrainer(KernelType* kernel, double negativeC, double positiveC, bool offset, bool unconstrained = false)
	: base_type(kernel,negativeC, positiveC, offset, unconstrained), m_computeDerivative(false), m_McSvmType(McSvm::WW) //make  Vapnik happy!
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CSvmTrainer"; }
	
	void setComputeBinaryDerivative(bool compute){
		m_computeDerivative = compute;
	}
	
	/// \brief sets the type of the multi-class svm used
	void setMcSvmType(McSvm type){
		m_McSvmType = type;
	}


	/// \brief Train the C-SVM.
	void train(KernelClassifier<InputType>& svm, LabeledData<InputType, unsigned int> const& dataset)
	{
		std::size_t classes = numberOfClasses(dataset);
		if(classes == 2){
			// prepare model
			std::size_t n = dataset.numberOfElements();
			auto& f = svm.decisionFunction();
			if (f.basis() == dataset.inputs() && f.kernel() == base_type::m_kernel && f.alpha().size1() == n && f.alpha().size2() == 1) {
				// warm start, keep the alphas (possibly clipped)
				if (this->m_trainOffset) f.offset() = RealVector(1);
				else f.offset() = RealVector();
			}
			else {
				f.setStructure(base_type::m_kernel, dataset.inputs(), this->m_trainOffset);
			}
			
			//dispatch to use the optimal implementation and solve the problem
			trainBinary(f,dataset);
			
			if (base_type::sparsify())
				f.sparsify();
			return;
		}
		//multiclass case: dispatch to the chosen Svm-type
		switch (m_McSvmType){
			case McSvm::WW:
				trainMc<detail::McSvmWWTrainer<InputType,CacheType> >(svm,dataset);
			break;
			case McSvm::CS:
				trainMc<detail::McSvmCSTrainer<InputType,CacheType> >(svm,dataset);
			break;
			case McSvm::LLW:
				trainMc<detail::McSvmLLWTrainer<InputType,CacheType> >(svm,dataset);
			break;
			case McSvm::ATM:
				trainMc<detail::McSvmATMTrainer<InputType,CacheType> >(svm,dataset);
			break;
			case McSvm::ATS:
				trainMc<detail::McSvmATSTrainer<InputType,CacheType> >(svm,dataset);
			break;
			case McSvm::ADM:
				trainMc<detail::McSvmADMTrainer<InputType,CacheType> >(svm,dataset);
			break;
			case McSvm::MMR:
				trainMc<detail::McSvmMMRTrainer<InputType,CacheType> >(svm,dataset);
			break;
			case McSvm::ReinforcedSvm:
				trainMc<detail::McReinforcedSvmTrainer<InputType,CacheType> >(svm,dataset);
			break;
			case McSvm::OVA://OVA is a special case and implemented here
				trainOVA(svm,dataset);
			break;
		}
	}

	/// \brief Train the C-SVM using weights.
	void train(KernelClassifier<InputType>& svm, WeightedLabeledData<InputType, unsigned int> const& dataset)
	{
		if(numberOfClasses(dataset) != 2)
			throw SHARKEXCEPTION("CSVM with weights is only implemented for binary problems");
		// prepare model
		std::size_t n = dataset.numberOfElements();
		auto& f = svm.decisionFunction();
		if (f.basis() == dataset.inputs() && f.kernel() == base_type::m_kernel && f.alpha().size1() == n && f.alpha().size2() == 1) {
			// warm start, keep the alphas
			if (this->m_trainOffset) f.offset() = RealVector(1);
			else f.offset() = RealVector();
		}
		else {
			f.setStructure(base_type::m_kernel, dataset.inputs(), this->m_trainOffset);
		}

		//dispatch to use the optimal implementation and solve the problem
		trainBinary(f, dataset);

		if (base_type::sparsify()) f.sparsify();
	}
	
	RealVector const& get_db_dParams()const{
		return m_db_dParams;
	}

private:
	
	template<class Trainer>
	void trainMc(KernelClassifier<InputType>& svm, LabeledData<InputType, unsigned int> const& dataset){
		Trainer trainer(base_type::m_kernel,this->C(),this->m_trainOffset);
		trainer.stoppingCondition() = this->stoppingCondition();
		trainer.precomputeKernel() = this->precomputeKernel();
		trainer.sparsify() = this->sparsify();
		trainer.shrinking() = this->shrinking();
		trainer.s2do() = this->s2do();
		trainer.verbosity() = this->verbosity();
		trainer.setCacheSize(this->cacheSize());
		trainer.train(svm,dataset);
		this->solutionProperties() = trainer.solutionProperties();
		base_type::m_accessCount = trainer.accessCount();
	}
	
	void trainOVA(KernelClassifier<InputType>& svm, const LabeledData<InputType, unsigned int>& dataset){
		std::size_t classes = numberOfClasses(dataset);
		svm.decisionFunction().setStructure(this->m_kernel,dataset.inputs(),this->m_trainOffset,classes);
		
		base_type::m_solutionproperties.type = QpNone;
		base_type::m_solutionproperties.accuracy = 0.0;
		base_type::m_solutionproperties.iterations = 0;
		base_type::m_solutionproperties.value = 0.0;
		base_type::m_solutionproperties.seconds = 0.0;
		for (unsigned int c=0; c<classes; c++)
		{
			LabeledData<InputType, unsigned int> bindata = oneVersusRestProblem(dataset, c);
			KernelClassifier<InputType> binsvm;
// TODO: maybe build the quadratic programs directly,
//       in order to profit from cached and
//       in particular from precomputed kernel
//       entries!
			CSvmTrainer<InputType, QpFloatType> bintrainer(base_type::m_kernel, this->C(),this->m_trainOffset);
			bintrainer.setCacheSize(this->cacheSize());
			bintrainer.sparsify() = false;
			bintrainer.stoppingCondition() = base_type::stoppingCondition();
			bintrainer.precomputeKernel() = base_type::precomputeKernel();		// sub-optimal!
			bintrainer.shrinking() = base_type::shrinking();
			bintrainer.s2do() = base_type::s2do();
			bintrainer.verbosity() = base_type::verbosity();
			bintrainer.train(binsvm, bindata);
			base_type::m_solutionproperties.iterations += bintrainer.solutionProperties().iterations;
			base_type::m_solutionproperties.seconds += bintrainer.solutionProperties().seconds;
			base_type::m_solutionproperties.accuracy = std::max(base_type::solutionProperties().accuracy, bintrainer.solutionProperties().accuracy);
			column(svm.decisionFunction().alpha(), c) = column(binsvm.decisionFunction().alpha(), 0);
			if (this->m_trainOffset)
				svm.decisionFunction().offset(c) = binsvm.decisionFunction().offset(0);
			base_type::m_accessCount += bintrainer.accessCount();
		}

		if (base_type::sparsify()) 
			svm.decisionFunction().sparsify();
	}
	
	//by default the normal unoptimized kernel matrix is used
	template<class T, class DatasetTypeT>
	void trainBinary(KernelExpansion<T>& svm, DatasetTypeT const& dataset){
		KernelMatrix<T, QpFloatType> km(*base_type::m_kernel, dataset.inputs());
		trainBinary(km,svm,dataset);
	}
	
	//in the case of a gaussian kernel and sparse vectors, we can use an optimized approach
	template<class T, class DatasetTypeT>
	void trainBinary(KernelExpansion<CompressedRealVector>& svm, DatasetTypeT const& dataset){
		//check whether a gaussian kernel is used
		typedef GaussianRbfKernel<CompressedRealVector> Gaussian;
		Gaussian const* kernel = dynamic_cast<Gaussian const*> (base_type::m_kernel);
		if(kernel != 0){//jep, use optimized kernel matrix
			GaussianKernelMatrix<CompressedRealVector,QpFloatType> km(kernel->gamma(),dataset.inputs());
			trainBinary(km,svm,dataset);
		}
		else{
			KernelMatrix<CompressedRealVector, QpFloatType> km(*base_type::m_kernel, dataset.inputs());
			trainBinary(km,svm,dataset);
		}
	}
	
	//create the problem for the unweighted datasets
	template<class Matrix, class T>
	void trainBinary(Matrix& km, KernelExpansion<T>& svm, LabeledData<T, unsigned int> const& dataset){
		if (QpConfig::precomputeKernel())
		{
			PrecomputedMatrix<Matrix> matrix(&km);
			CSVMProblem<PrecomputedMatrix<Matrix> > svmProblem(matrix,dataset.labels(),base_type::m_regularizers);
			optimize(svm,svmProblem,dataset);
		}
		else
		{
			CachedMatrix<Matrix> matrix(&km);
			CSVMProblem<CachedMatrix<Matrix> > svmProblem(matrix,dataset.labels(),base_type::m_regularizers);
			optimize(svm,svmProblem,dataset);
		}
		base_type::m_accessCount = km.getAccessCount();
	}
	
	// create the problem for the weighted datasets
	template<class Matrix, class T>
	void trainBinary(Matrix& km, KernelExpansion<T>& svm, WeightedLabeledData<T, unsigned int> const& dataset){
		if (QpConfig::precomputeKernel())
		{
			PrecomputedMatrix<Matrix> matrix(&km);
			GeneralQuadraticProblem<PrecomputedMatrix<Matrix> > svmProblem(
				matrix,dataset.labels(),dataset.weights(),base_type::m_regularizers
			);
			optimize(svm,svmProblem,dataset.data());
		}
		else
		{
			CachedMatrix<Matrix> matrix(&km);
			GeneralQuadraticProblem<CachedMatrix<Matrix> > svmProblem(
				matrix,dataset.labels(),dataset.weights(),base_type::m_regularizers
			);
			optimize(svm,svmProblem,dataset.data());
		}
		base_type::m_accessCount = km.getAccessCount();
	}

private:
	
	template<class SVMProblemType>
	void optimize(KernelExpansion<InputType>& svm, SVMProblemType& svmProblem, LabeledData<InputType, unsigned int> const& dataset){
		if (this->m_trainOffset)
		{
			typedef SvmShrinkingProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem,base_type::m_shrinking);
			QpSolver< ProblemType > solver(problem);
			// truncate the existing solution to the bounds
			RealVector const& reg = this->regularizationParameters();
			double C_minus = reg(0);
			double C_plus = (reg.size() == 1) ? reg(0) : reg(1);
			std::size_t i=0;
			for (auto label : dataset.labels().elements()) {
				double a = svm.alpha()(i, 0);
				if (label == 0) a = std::max(std::min(a, 0.0), -C_minus);
				else            a = std::min(std::max(a, 0.0), C_plus);
				svm.alpha()(i, 0) = a;
				i++;
			}
			problem.setInitialSolution(blas::column(svm.alpha(), 0));
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			column(svm.alpha(),0)= problem.getUnpermutedAlpha();
			svm.offset(0) = computeBias(problem,dataset);
		}
		else
		{
			typedef BoxConstrainedShrinkingProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem,base_type::m_shrinking);
			QpSolver< ProblemType> solver(problem);
			// truncate the existing solution to the bounds
			RealVector const& reg = this->regularizationParameters();
			double C_minus = reg(0);
			double C_plus = (reg.size() == 1) ? reg(0) : reg(1);
			std::size_t i=0;
			for (auto label : dataset.labels().elements()) {
				double a = svm.alpha()(i, 0);
				if (label == 0) a = std::max(std::min(a, 0.0), -C_minus);
				else            a = std::min(std::max(a, 0.0), C_plus);
				svm.alpha()(i, 0) = a;
				i++;
			}
			problem.setInitialSolution(blas::column(svm.alpha(), 0));
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			column(svm.alpha(),0) = problem.getUnpermutedAlpha();
		}
	}
	
	RealVector m_db_dParams; ///< in the rare case that there are only bounded SVs and no free SVs, this will hold the derivative of b w.r.t. the hyperparameters. Derivative w.r.t. C is last.

	bool m_computeDerivative;
	McSvm m_McSvmType;

	template<class Problem>
	double computeBias(Problem const& problem, LabeledData<InputType, unsigned int> const& dataset){
		std::size_t nkp = base_type::m_kernel->numberOfParameters();
		m_db_dParams.resize(nkp+1);
		m_db_dParams.clear();

		std::size_t ic = problem.dimensions();
		if (ic == 0) return 0.0;

		// compute the offset from the KKT conditions
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		std::size_t freeVars = 0;
		std::size_t lower_i = 0;
		std::size_t upper_i = 0;
		for (std::size_t i=0; i<ic; i++)
		{
			double value = problem.gradient(i);
			if (problem.alpha(i) == problem.boxMin(i))
			{
				if (value > lowerBound) { //in case of no free SVs, we are looking for the largest gradient of all alphas at the lower bound
					lowerBound = value;
					lower_i = i;
				}
			}
			else if (problem.alpha(i) == problem.boxMax(i))
			{
				if (value < upperBound) { //in case of no free SVs, we are looking for the smallest gradient of all alphas at the upper bound
					upperBound = value;
					upper_i = i;
				}
			}
			else
			{
				sum += value;
				freeVars++;
			}
		}
		if (freeVars > 0)
			return sum / freeVars;		//stabilized (averaged) exact value

		if(!m_computeDerivative)
			return 0.5 * (lowerBound + upperBound);	//best estimate

		lower_i = problem.permutation(lower_i);
		upper_i = problem.permutation(upper_i);

		SHARK_CHECK(base_type::m_regularizers.size() == 1, "derivative only implemented for SVM with one C" );

		// We next compute the derivative of lowerBound and upperBound wrt C, in order to then get that of b wrt C.
		// The equation at the foundation of this simply is g_i = y_i - \sum_j \alpha_j K_{ij} .
		double dlower_dC = 0.0;
		double dupper_dC = 0.0;
		// At the same time, we also compute the derivative of lowerBound and upperBound wrt the kernel parameters.
		// The equation at the foundation of this simply is g_i = y_i - \sum_j \alpha_j K_{ij} .
		RealVector dupper_dkernel( nkp,0 );
		RealVector dlower_dkernel( nkp,0 );
		//state for eval and evalDerivative of the kernel
		boost::shared_ptr<State> kernelState = base_type::m_kernel->createState();
		RealVector der(nkp ); //derivative storage helper
		//todo: O.K.: here kernel single input derivative would be usefull
		//also it can be usefull to use here real batch processing and use batches of size 1 for lower /upper
		//and instead of singleInput whole batches.
		//what we do is, that we use the batched input versions with batches of size one.
		typename Batch<InputType>::type singleInput = Batch<InputType>::createBatch( dataset.element(0).input, 1 );
		typename Batch<InputType>::type lowerInput = Batch<InputType>::createBatch( dataset.element(lower_i).input, 1 );
		typename Batch<InputType>::type upperInput = Batch<InputType>::createBatch( dataset.element(upper_i).input, 1 );
		get( lowerInput, 0 ) = dataset.element(lower_i).input; //copy the current input into the batch
		get( upperInput, 0 ) = dataset.element(upper_i).input; //copy the current input into the batch
		RealMatrix one(1,1,1); //weight of input
		RealMatrix result(1,1); //stores the result of the call

		for (std::size_t i=0; i<ic; i++) {
			double cur_alpha = problem.alpha(problem.permutation(i));
			if ( cur_alpha != 0 ) {
				int cur_label = ( cur_alpha>0.0 ? 1 : -1 );
				get( singleInput, 0 ) = dataset.element(i).input; //copy the current input into the batch
				// treat contributions of largest gradient at lower bound
				base_type::m_kernel->eval( lowerInput, singleInput, result, *kernelState );
				dlower_dC += cur_label * result(0,0);
				base_type::m_kernel->weightedParameterDerivative( lowerInput, singleInput,one, *kernelState, der );
				for ( std::size_t k=0; k<nkp; k++ ) {
					dlower_dkernel(k) += cur_label * der(k);
				}
				// treat contributions of smallest gradient at upper bound
				base_type::m_kernel->eval( upperInput, singleInput,result, *kernelState );
				dupper_dC += cur_label * result(0,0);
				base_type::m_kernel->weightedParameterDerivative( upperInput, singleInput, one, *kernelState, der );
				for ( std::size_t k=0; k<nkp; k++ ) {
					dupper_dkernel(k) += cur_label * der(k);
				}
			}
		}
		// assign final values to derivative of b wrt hyperparameters
		m_db_dParams( nkp ) = -0.5 * ( dlower_dC + dupper_dC );
		for ( std::size_t k=0; k<nkp; k++ ) {
			m_db_dParams(k) = -0.5 * this->C() * ( dlower_dkernel(k) + dupper_dkernel(k) );
		}
		if ( base_type::m_unconstrained ) {
			m_db_dParams( nkp ) *= this->C();
		}
		
		return 0.5 * (lowerBound + upperBound);	//best estimate
	}
};


template <class InputType>
class LinearCSvmTrainer : public AbstractLinearSvmTrainer<InputType>
{
public:
	typedef AbstractLinearSvmTrainer<InputType> base_type;

	LinearCSvmTrainer(double C, bool offset, bool unconstrained = false) 
	: AbstractLinearSvmTrainer<InputType>(C, offset, unconstrained){}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearCSvmTrainer"; }
	
	/// \brief sets the type of the multi-class svm used
	void setMcSvmType(McSvm type){
		m_McSvmType = type;
	}

	void train(LinearClassifier<InputType>& model, LabeledData<InputType, unsigned int> const& dataset)
	{
		std::size_t classes = numberOfClasses(dataset);
		if(classes == 2){
			trainBinary(model,dataset);
			return;
		}
		switch (m_McSvmType){
			case McSvm::WW:
				trainMc<QpMcLinearWW<InputType> >(model,dataset,classes);
			break;
			case McSvm::CS:
				trainMc<QpMcLinearCS<InputType> >(model,dataset,classes);
			break;
			case McSvm::LLW:
				trainMc<QpMcLinearLLW<InputType> >(model,dataset,classes);
			break;
			case McSvm::ATM:
				trainMc<QpMcLinearATM<InputType> >(model,dataset,classes);
			break;
			case McSvm::ATS:
				trainMc<QpMcLinearATS<InputType> >(model,dataset,classes);
			break;
			case McSvm::ADM:
				trainMc<QpMcLinearADM<InputType> >(model,dataset,classes);
			break;
			case McSvm::MMR:
				trainMc<QpMcLinearMMR<InputType> >(model,dataset,classes);
			break;
			case McSvm::ReinforcedSvm:
				trainMc<QpMcLinearReinforced<InputType> >(model,dataset,classes);
			break;
			case McSvm::OVA://OVA is a special case and implemented here
				trainOVA(model,dataset,classes);
			break;
		}
	}
private:
	McSvm m_McSvmType;

	void trainBinary(LinearClassifier<InputType>& model, LabeledData<InputType, unsigned int> const& dataset)
	{
		std::size_t dim = inputDimension(dataset);
		QpBoxLinear<InputType> solver(dataset, dim);
		solver.solve(
				base_type::C(),
				0.0,
				QpConfig::stoppingCondition(),
				&QpConfig::solutionProperties(),
				QpConfig::verbosity() > 0);
		
		if(!this->trainOffset()){
			RealMatrix w(1, dim, 0.0);
			row(w,0) = solver.solutionWeightVector();
			model.decisionFunction().setStructure(w);
			return;
		}
		
		double offset = 0;
		double stepSize = 0.1;
		double grad = solver.offsetGradient();
		while(stepSize > 0.1*QpConfig::stoppingCondition().minAccuracy){
			offset+= (grad < 0? -stepSize:stepSize);
			solver.setOffset(offset);
			solver.solve(
				base_type::C(),
				0.0,
				QpConfig::stoppingCondition(),
				&QpConfig::solutionProperties(),
				QpConfig::verbosity() > 0);
			double newGrad = solver.offsetGradient();
			if(newGrad == 0)
				break;
			if(newGrad*grad < 0)
				stepSize *= 0.5;
			else
				stepSize *= 1.6;
			grad = newGrad;
		}
		
		RealMatrix w(1, dim, 0.0);
		noalias(row(w,0)) = solver.solutionWeightVector();
		model.decisionFunction().setStructure(w,RealVector(1,offset));
		
	}
	template<class Solver>
	void trainMc(LinearClassifier<InputType>& model, LabeledData<InputType, unsigned int> const& dataset, std::size_t classes){
		std::size_t dim = inputDimension(dataset);

		Solver solver(dataset, dim, classes);
		RealMatrix w = solver.solve(this->C(), this->stoppingCondition(), &this->solutionProperties(), this->verbosity() > 0);
		model.decisionFunction().setStructure(w);
	}
	
	void trainOVA(LinearClassifier<InputType>& model, const LabeledData<InputType, unsigned int>& dataset, std::size_t classes)
	{
		base_type::m_solutionproperties.type = QpNone;
		base_type::m_solutionproperties.accuracy = 0.0;
		base_type::m_solutionproperties.iterations = 0;
		base_type::m_solutionproperties.value = 0.0;
		base_type::m_solutionproperties.seconds = 0.0;

		std::size_t dim = inputDimension(dataset);
		RealMatrix w(classes, dim);
		for (unsigned int c=0; c<classes; c++)
		{
			LabeledData<InputType, unsigned int> bindata = oneVersusRestProblem(dataset, c);
			QpBoxLinear<InputType> solver(bindata, dim);
			QpSolutionProperties prop;
			solver.solve(this->C(), 0.0, base_type::m_stoppingcondition, &prop, base_type::m_verbosity > 0);
			noalias(row(w, c)) = solver.solutionWeightVector();
			base_type::m_solutionproperties.iterations += prop.iterations;
			base_type::m_solutionproperties.seconds += prop.seconds;
			base_type::m_solutionproperties.accuracy = std::max(base_type::solutionProperties().accuracy, prop.accuracy);
		}
		model.decisionFunction().setStructure(w);
	}
};


template <class InputType, class CacheType = float>
class SquaredHingeCSvmTrainer : public AbstractSvmTrainer<InputType, unsigned int>
{
public:
	typedef CacheType QpFloatType;

	typedef RegularizedKernelMatrix<InputType, QpFloatType> KernelMatrixType;
	typedef CachedMatrix< KernelMatrixType > CachedMatrixType;
	typedef PrecomputedMatrix< KernelMatrixType > PrecomputedMatrixType;

	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, unsigned int> base_type;

	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver??
	SquaredHingeCSvmTrainer(KernelType* kernel, double C, bool unconstrained = false)
	: base_type(kernel, C, unconstrained)
	{ }
	
	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  negativeC   regularization parameter of the negative class (label 0)
	//! \param  positiveC    regularization parameter of the positive class (label 1)
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	SquaredHingeCSvmTrainer(KernelType* kernel, double negativeC, double positiveC, bool unconstrained = false)
	: base_type(kernel,negativeC, positiveC, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SquaredHingeCSvmTrainer"; }

	/// \brief Train the C-SVM.
	void train(KernelClassifier<InputType>& svm, LabeledData<InputType, unsigned int> const& dataset)
	{		
		svm.decisionFunction().setStructure(base_type::m_kernel,dataset.inputs(),this->m_trainOffset);
		
		RealVector diagonalModifier(dataset.numberOfElements(),0.5/base_type::m_regularizers(0));
		if(base_type::m_regularizers.size() != 1){
			for(std::size_t i = 0; i != diagonalModifier.size();++i){
				diagonalModifier(i) = 0.5/base_type::m_regularizers(dataset.element(i).label);
			}
		}
		
		KernelMatrixType km(*base_type::m_kernel, dataset.inputs(),diagonalModifier);
		if (QpConfig::precomputeKernel())
		{
			PrecomputedMatrixType matrix(&km);
			optimize(svm.decisionFunction(),matrix,diagonalModifier,dataset);
		}
		else
		{
			CachedMatrixType matrix(&km);
			optimize(svm.decisionFunction(),matrix,diagonalModifier,dataset);
		}
		base_type::m_accessCount = km.getAccessCount();
		if (base_type::sparsify()) svm.decisionFunction().sparsify();

	}

private:
	
	template<class Matrix>
	void optimize(KernelExpansion<InputType>& svm, Matrix& matrix,RealVector const& diagonalModifier, LabeledData<InputType, unsigned int> const& dataset){
		typedef CSVMProblem<Matrix> SVMProblemType;
		SVMProblemType svmProblem(matrix,dataset.labels(),1e100);
		if (this->m_trainOffset)
		{
			typedef SvmShrinkingProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem,base_type::m_shrinking);
			QpSolver< ProblemType > solver(problem);
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			column(svm.alpha(),0)= problem.getUnpermutedAlpha();
			//compute the bias
			double sum = 0.0;
			std::size_t freeVars = 0;
			for (std::size_t i=0; i < problem.dimensions(); i++)
			{
				if(problem.alpha(i) > problem.boxMin(i) && problem.alpha(i) < problem.boxMax(i)){
					sum += problem.gradient(i) - problem.alpha(i)*2*diagonalModifier(i);
					freeVars++;
				}
			}
			if (freeVars > 0)
				svm.offset(0) =  sum / freeVars;		//stabilized (averaged) exact value
			else
				svm.offset(0) = 0;
		}
		else
		{
			typedef BoxConstrainedShrinkingProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem,base_type::m_shrinking);
			QpSolver< ProblemType > solver(problem);
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			column(svm.alpha(),0) = problem.getUnpermutedAlpha();
			
		}
	}
};


template <class InputType>
class SquaredHingeLinearCSvmTrainer : public AbstractLinearSvmTrainer<InputType>
{
public:
	typedef AbstractLinearSvmTrainer<InputType> base_type;

	SquaredHingeLinearCSvmTrainer(double C, bool unconstrained = false) 
	: AbstractLinearSvmTrainer<InputType>(C, false, unconstrained){}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SquaredHingeLinearCSvmTrainer"; }

	void train(LinearClassifier<InputType>& model, LabeledData<InputType, unsigned int> const& dataset)
	{
		std::size_t dim = inputDimension(dataset);
		QpBoxLinear<InputType> solver(dataset, dim);
		RealMatrix w(1, dim, 0.0);
		solver.solve(
				1e100,
				1.0 / base_type::C(),
				QpConfig::stoppingCondition(),
				&QpConfig::solutionProperties(),
				QpConfig::verbosity() > 0);
		row(w,0) = solver.solutionWeightVector();
		model.decisionFunction().setStructure(w);
	}
};


}
#endif
