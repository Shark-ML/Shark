#ifndef SHARK_ML_MODEL_LINEARCLASSIFIER_H
#define SHARK_ML_MODEL_LINEARCLASSIFIER_H

#include <shark/Models/LinearModel.h>
#include <shark/Models/Converter.h>
namespace shark {

/*! \brief Basic linear classifier.
 *
 *  The LinearClassifier class is a multi class classifier model
 *  suited for linear discriminant analysis. For c classes
 *  \f$ 0, \dots, c-1 \f$  the model computes
 *   
 *  \f$ \arg \max_i w_i^T x + b_i \f$
 *  
 *  Thus is it a linear model with arg max computation.
 *  The internal linear model can be queried using decisionFunction().
 */ 
template<class VectorType = RealVector>
class LinearClassifier : public ArgMaxConverter<LinearModel<VectorType> >
{
public:
	LinearClassifier(){}

	std::string name() const
	{ return "LinearClassifier"; }
};
}
#endif
