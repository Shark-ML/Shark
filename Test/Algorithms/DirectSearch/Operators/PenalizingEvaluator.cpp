#define BOOST_TEST_MODULE DIRECTSEARCH_PENALIZING_EVALUATOR
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Evaluation/PenalizingEvaluator.h>
#include <shark/ObjectiveFunctions/Benchmarks/ZDT1.h>
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>

using namespace shark;


struct TestIndividualMOO{
	RealVector m_point;
	RealVector m_penalizedFitness;
	RealVector m_unpenalizedFitness;
	
	RealVector& penalizedFitness(){ return m_penalizedFitness; }
	RealVector& unpenalizedFitness(){ return m_unpenalizedFitness; }
	RealVector const& searchPoint()const{ return m_point; }
};

struct TestIndividualSOO{
	RealVector m_point;
	double m_penalizedFitness;
	double m_unpenalizedFitness;
	
	double& penalizedFitness(){ return m_penalizedFitness; }
	double& unpenalizedFitness(){ return m_unpenalizedFitness; }
	RealVector const& searchPoint()const{ return m_point; }
};

struct SOOTestFunction  : public SingleObjectiveFunction
{
	
	SOOTestFunction(std::size_t numVariables) :  m_rosenbrock(numVariables), m_handler(numVariables,0,1) {
		announceConstraintHandler(&m_handler);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SOOTestFunction"; }
	
	std::size_t numberOfVariables()const{
		return m_rosenbrock.numberOfVariables();
	}

	ResultType eval( const SearchPointType & x ) const {
		return m_rosenbrock.eval(x);
	}

private:
	Rosenbrock m_rosenbrock;
	BoxConstraintHandler<SearchPointType> m_handler;
};

//check that feasible points are not penalized
BOOST_AUTO_TEST_SUITE (Algorithms_DirectSearch_Operators_PenalizingEvaluator)

BOOST_AUTO_TEST_CASE( PenalizingEvaluator_SingleObjective_Feasible ) {
	PenalizingEvaluator evaluator;
	evaluator.m_penaltyFactor = 1000;//make errors obvious!
	SOOTestFunction objective(10);
	for(std::size_t i = 0; i != 1000; ++i){
		TestIndividualSOO tester;
		objective.proposeStartingPoint(tester.m_point);
		
		BOOST_REQUIRE(objective.isFeasible(tester.m_point));
		double fitness = objective(tester.m_point);
		
		evaluator(objective,tester);
		
		BOOST_CHECK_EQUAL(fitness,tester.m_penalizedFitness);
		BOOST_CHECK_EQUAL(fitness,tester.m_unpenalizedFitness);
	}
}

//check that infeasible points in the single objective case are penalized correctly
BOOST_AUTO_TEST_CASE( PenalizingEvaluator_SingleObjective_Infeasible ) {
	BoxConstraintHandler<RealVector> generator(10,-5,5);
	PenalizingEvaluator evaluator;
	SOOTestFunction objective(10);
	//over 1000 infeasible points (we skip feasible ones)
	std::size_t trials = 0;
	while(trials < 1000){
		TestIndividualSOO tester;
		generator.generateRandomPoint(tester.m_point);
		
		RealVector corrected=tester.m_point;
		objective.closestFeasible(corrected);
		BOOST_REQUIRE(objective.isFeasible(corrected));
		
		double fitness = objective(corrected);
		evaluator.m_penaltyFactor = Rng::uni(0.1,1);//make errors obvious!
		evaluator(objective,tester);
		
		//calculate the expected penalty
		double expectedPenalty = evaluator.m_penaltyFactor*norm_sqr(tester.m_point-corrected);
		BOOST_CHECK_CLOSE(fitness,tester.m_unpenalizedFitness,1.e-10);
		BOOST_CHECK_CLOSE(tester.m_penalizedFitness-fitness,expectedPenalty, 1.e-10);
		++trials;
	}
}

//check that feasible points are not penalized
BOOST_AUTO_TEST_CASE( PenalizingEvaluator_MultiObjective_Feasible ) {
	PenalizingEvaluator evaluator;
	evaluator.m_penaltyFactor = 1000;//make errors obvious!
	ZDT1 objective(10);
	for(std::size_t i = 0; i != 1000; ++i){
		TestIndividualMOO tester;
		objective.proposeStartingPoint(tester.m_point);
		
		BOOST_REQUIRE(objective.isFeasible(tester.m_point));
		RealVector fitness = objective(tester.m_point);
		
		evaluator(objective,tester);
		BOOST_REQUIRE_EQUAL(tester.m_penalizedFitness.size(), fitness.size());
		BOOST_REQUIRE_EQUAL(tester.m_unpenalizedFitness.size(), fitness.size());
		
		for(std::size_t i = 0; i != fitness.size(); ++i){
			BOOST_CHECK_EQUAL(fitness(i),tester.m_penalizedFitness(i));
			BOOST_CHECK_EQUAL(fitness(i),tester.m_unpenalizedFitness(i));
		}
	}
}

//check that infeasible points in the multi objective case are penalized correctly
BOOST_AUTO_TEST_CASE( PenalizingEvaluator_MultiObjective_Infeasible ) {
	BoxConstraintHandler<RealVector> generator(10,-5,5);
	PenalizingEvaluator evaluator;
	ZDT1 objective(10);
	//over 1000 infeasible points (we skip feasible ones)
	std::size_t trials = 0;
	while(trials < 1000){
		TestIndividualMOO tester;
		generator.generateRandomPoint(tester.m_point);
		if(objective.isFeasible(tester.m_point)) continue;//don't increase trial counters
		
		RealVector corrected=tester.m_point;
		objective.closestFeasible(corrected);
		BOOST_REQUIRE(objective.isFeasible(corrected));
		
		RealVector fitness = objective(corrected);
		evaluator.m_penaltyFactor = Rng::uni(0.1,1);//make errors obvious!
		evaluator(objective,tester);
		BOOST_REQUIRE_EQUAL(tester.m_penalizedFitness.size(), fitness.size());
		BOOST_REQUIRE_EQUAL(tester.m_unpenalizedFitness.size(), fitness.size());
		
		//calculate the expected penalty on every dimension
		double expectedPenalty = evaluator.m_penaltyFactor*norm_sqr(tester.m_point-corrected);
		
		for(std::size_t i = 0; i != fitness.size(); ++i){
			BOOST_CHECK_CLOSE(fitness(i),tester.m_unpenalizedFitness(i),1.e-10);
			BOOST_CHECK_CLOSE(tester.m_penalizedFitness(i)-fitness(i),expectedPenalty, 1.e-10);
		}
		++trials;
	}
}
BOOST_AUTO_TEST_SUITE_END()
