#ifndef SHARK_ML_MODEL_LINEARCLASSIFIER_H
#define SHARK_ML_MODEL_LINEARCLASSIFIER_H

#include <shark/Models/AbstractModel.h>
namespace shark {

/*! \brief Basic linear classifier.
 *
 *  The LinearClassifier class is a multi class classifier model
 *  suited for linear discriminant analysis. For c classes
 *  \f$ 0, \dots, c-1 \f$ the model holds class mean vectors
 *  \f$ m_c \f$ and a shared data scatter matrix \f$ M \f$. It
 *  predicts the class of a vector x according to the rule
 *  \f$ \textrm{argmin}_{c} (x - m_c)^T M^{-1} (x - m_c) \f$.
 *  The output is an integer in the range [0,c-1] telling to which class the point belongs
 * 
 *  As default, this class uses the class means and the inverse
 *  covariance matrix. So these are also the parameters the
 *  vector returned by #parameterVector consists of.
 *  However there exists a convenience function
 *  #importCovarianceMatrix which automatically inverts the covariance
 *  matrix for use of the class. Be warned, that this may take a lot
 *  of time
 */ 
class LinearClassifier : public AbstractModel<RealVector,unsigned int>
{
protected:
	RealMatrix m_inverseCholesky;
	RealMatrix m_means;
	RealMatrix m_transformedMeans;
	RealVector m_classBias;
	
	//true if the model has been changed since the last call to eval
	bool m_changed;
public:
	LinearClassifier():m_changed(false){
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearClassifier"; }

	void setStructure(std::size_t inputSize,std::size_t classes);

	void configure( PropertyTree const& node );

	std::size_t inputSize()const{
		return m_inverseCholesky.size1();
	}
	std::size_t outputSize()const{
		return m_means.size1();
	}
	std::size_t numberOfClasses()const{
		return m_means.size1();
	}
	
	RealVector parameterVector()const;
	void setParameterVector(RealVector const& newParameters);

	std::size_t numberOfParameters()const;

	void setInverseCholeskyMatrix(const RealMatrix& matrix){
		m_inverseCholesky = matrix;
	}
	const RealMatrix& inverseCholeskyMatrix()const{
		return m_inverseCholesky;
	}

	void importCovarianceMatrix(RealMatrix const& covariance);

	void setClassMean(std::size_t classID, RealVector const& mean);
	ConstRealMatrixRow classMean(std::size_t classID)const{
		SIZE_CHECK(classID<numberOfClasses());
		return row(m_means,classID);
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}
	using AbstractModel<RealVector,unsigned int >::eval;
	void eval(BatchInputType const& patterns, BatchOutputType& outputs)const;
	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const{
		eval(patterns,outputs);
	}

	/// From ISerializable, reads a model from an archive
	void read( InArchive & archive );

	/// From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const;

	/// Return  memberships to the different classes, the memberships sum to unity
	Data<RealVector> softMembership(Data<InputType> const & patterns) const;

	/// Compute memberships to the different classes, the memberships sum to unity
	void softMembership(BatchInputType const& patterns, Batch<RealVector>::type& output) const;
};
}
#endif
