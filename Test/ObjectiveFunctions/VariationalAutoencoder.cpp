#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/VariationalAutoencoderError.h>
#include <shark/Models/LinearModel.h>
#include <shark/Data/Dataset.h>

#define BOOST_TEST_MODULE ObjFunct_ErrorFunction
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;


class ZeroLoss : public AbstractLoss<RealVector,RealVector>
{
public:
	ZeroLoss(){this->m_features|=HAS_FIRST_DERIVATIVE;}
	std::string name() const
	{ return "ZeroLoss"; }

	using base_type::eval;
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const {
		return 0.0;
	}
	double evalDerivative(BatchLabelType const& label, BatchOutputType const& prediction, BatchOutputType& gradient) const {
		gradient.resize(prediction.size1(),prediction.size2());
		gradient.clear();
		return 0.0;
	}
};


BOOST_AUTO_TEST_SUITE (ObjectiveFunctions_VariationalAutoencoder)

//when setting the loss to constant 0, we only get the KL-part, which is deterministic
BOOST_AUTO_TEST_CASE( ObjFunct_VariationalAutoencoder_Grad_NoLoss )
{
	std::size_t trainExamples = 100;

	// create samples
	std::vector<RealVector> input(trainExamples,RealVector(2));
	for (size_t i=0;i!=trainExamples;++i) {
		input[i](0) = random::gauss(random::globalRng, 0,4);
		input[i](1) = random::uni(random::globalRng, 0, 9);
	}
	UnlabeledData<RealVector> dataset = createUnlabeledDataFromRange(input,trainExamples);
	
	ZeroLoss loss;
	LinearModel<> encoder(2,4,false);
	LinearModel<> decoder(2,2,false);
	initRandomNormal(encoder,1);
	initRandomNormal(decoder,1);
	VariationalAutoencoderError<RealVector> error(dataset, &encoder, &decoder,&loss);

	RealVector gradEst(error.numberOfVariables(),0.0);
	RealVector p = error. proposeStartingPoint();
	double eps = 0.01;
	for(std::size_t i = 0; i != p.size(); ++i){
		RealVector pminus = p;
		pminus(i) -= eps;
		RealVector pplus = p;
		pplus(i) += eps;
		gradEst(i) = error.eval(pplus);
		gradEst(i) -= error.eval(pminus);			
	}
	gradEst /= 2*eps;
	
	RealVector grad(gradEst.size());
	error.evalDerivative(p,grad);
	for(std::size_t i = 0; i != p.size(); ++i){
		BOOST_CHECK_CLOSE(grad(i),gradEst(i),1.0);
	}
	std::cout<<gradEst<<std::endl;
	std::cout<<grad<<std::endl;
}

BOOST_AUTO_TEST_CASE( ObjFunct_VariationalAutoencoder_Grad_Full )
{
	std::size_t trainExamples = 100;

	// create samples
	std::vector<RealVector> input(trainExamples,RealVector(2));
	for (size_t i=0;i!=trainExamples;++i) {
		input[i](0) = random::gauss(random::globalRng, 0,4);
		input[i](1) = random::uni(random::globalRng, 0, 9);
	}
	UnlabeledData<RealVector> dataset = createUnlabeledDataFromRange(input,trainExamples);
	
	SquaredLoss<> loss;
	LinearModel<> encoder(2,4,false);
	LinearModel<> decoder(2,2,false);
	initRandomNormal(encoder,1);
	initRandomNormal(decoder,1);
	VariationalAutoencoderError<RealVector> error(dataset, &encoder, &decoder,&loss);

	RealVector gradEst(error.numberOfVariables(),0.0);
	RealVector p = error. proposeStartingPoint();
	double eps = 0.01;
	for(std::size_t i = 0; i != p.size(); ++i){
		RealVector pminus = p;
		pminus(i) -= eps;
		RealVector pplus = p;
		pplus(i) += eps;
		
		random::globalRng.seed(42);
		gradEst(i) += error.eval(pplus);
		random::globalRng.seed(42);
		gradEst(i) -= error.eval(pminus);		
	}
	gradEst /= 2*eps;
	
	random::globalRng.seed(42);
	RealVector grad(gradEst.size());
	error.evalDerivative(p,grad);
	for(std::size_t i = 0; i != p.size(); ++i){
		BOOST_CHECK_CLOSE(grad(i),gradEst(i),1.0);
	}
}


BOOST_AUTO_TEST_SUITE_END()
