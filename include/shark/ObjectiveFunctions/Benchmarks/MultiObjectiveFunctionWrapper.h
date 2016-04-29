//===========================================================================
/*!
 * 
 *
 * \author     Oswin Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_MULTIOBJECTIVEFUNCTIONWRAPPER_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_MULTIOBJECTIVEFUNCTIONWRAPPER_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/Benchmarks/RotatedErrorFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Rng/GlobalRng.h>

#include <shark/LinAlg/rotations.h>
#include <tuple>

namespace shark {
/// \brief Wraps a set of single objective functions and combines them to a multi-objective function
///
/// The function requires that all underlying functions are unconstrained.
/// Further calling init() initializes a random translation vector for each function.
/// This ensures that for eample if all functions have their optimum at 0
/// afterwards they all have a different optimum and thus the front is non-degenerate
class MultiObjectiveFunctionWrapper : public MultiObjectiveFunction{
public:
	MultiObjectiveFunctionWrapper(){
		m_features |= CAN_PROPOSE_STARTING_POINT;
	}
	
	/// \brief Add an objective to this
	///
	/// A pointer to the function is stored and used for evaluation.
	/// The function is not allowed to be constrained and it must have the same number of
	/// variables as all previously included functions.
	///
	/// All functions must be added before calling init()
	void addObjective(SingleObjectiveFunction* f){
		SHARK_CHECK(f != nullptr, "[MultiObjectiveFunctionWrapper ::addFunction] f is not allowed to be 0");
		SHARK_CHECK(m_functions.empty() || numberOfVariables() == f->numberOfVariables(), "[MultiObjectiveFunctionWrapper ::addFunction] f has different number of variables");
		SHARK_CHECK(!f->isConstrained(), "[MultiObjectiveFunctionWrapper ::addFunction] f is not allowed to be constrained");
		m_functions.push_back(f);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "MultiObjectiveFunctionWrapper"; }

	std::size_t numberOfObjectives()const{
		return m_functions.size();
	}
	
	std::size_t numberOfVariables()const{
		return m_functions.empty()? 0: m_functions[0]->numberOfVariables();
	}

	void init() {
		m_translations.clear();
		
		for(auto* f: m_functions)
		{
			RealVector translation(numberOfVariables());
			for(double& v: translation){
				v=Rng::uni(0,2);
			}
			m_translations.push_back(translation);
			f->init();
		}
	}

	ResultType eval( SearchPointType const& x ) const {
		m_evaluationCounter++;

		ResultType value( numberOfObjectives() );
		for(std::size_t  i = 0; i != value.size(); ++i){
			value(i) = m_functions[i]->eval( x - m_translations[i]);
		}
		return value;
	}

	SearchPointType proposeStartingPoint() const {
		RealVector x(numberOfVariables());

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = Rng::uni(-10,10);
		}
		return x;
	}

private:
	std::vector<SingleObjectiveFunction*> m_functions;
	std::vector<RealVector> m_translations;
};


namespace detail{
	//taken from the web. implements an std::integer_sequence type representing a sequence 0,...,N-1
	template<int...> struct integer_sequence { using type = integer_sequence; };
	template<typename T1, typename T2> struct integer_sequence_concat;
	template<int... I1, int... I2> struct integer_sequence_concat<integer_sequence<I1...>, integer_sequence<I2...>>: integer_sequence<I1..., (sizeof...(I1) + I2)...> {};

	//generate_integer_sequence gener
	template<int N> struct generate_integer_sequence;
	template<int N> struct generate_integer_sequence: integer_sequence_concat<typename generate_integer_sequence<N/2>::type, typename generate_integer_sequence<N-N/2>::type>::type {};
	template <> struct generate_integer_sequence<0>: integer_sequence<>{};
	template <> struct generate_integer_sequence<1>: integer_sequence<0>{};
};


//todo there is a general implementation based on parameter packs. 

/// \brief Creates  a multi-objective Benchmark from a set of given single objective functions
///
/// A variadic number template is used to generate a set of benchmarks.
/// eg MultiObjectiveBenchmark<Sphere,Ellispoid,Rosenbrock> sets up a three-objective Benchmark.
///
/// A random rotation and translation is applie to each benchmark, thus
/// MultiObjectiveBenchmark<Sphere,Sphere> forms a non-degenerate front.
/// the ith objective can be queried via the get<i> member function.
template<class ... Objectives>
class MultiObjectiveBenchmark: public MultiObjectiveFunctionWrapper{
public:
	MultiObjectiveBenchmark(std::size_t numVariables = 5){
		setupRotations(typename detail::generate_integer_sequence<sizeof...(Objectives)>::type());
		setNumberOfVariables(numVariables);//prevent that different objectives have different default number of variables.
		for(auto& rotation: m_rotations){
			addObjective(&rotation);
		}
	};
	
	bool hasScalableDimensionality()const{
		return true;
	}
	
	void setNumberOfVariables( std::size_t numberOfVariables ){
		for(auto& rotation: m_rotations){
			rotation.setNumberOfVariables(numberOfVariables);
		}
	}
	
	template<int N>
	typename std::tuple_element<N, std::tuple<Objectives...> >::type get(){
		return std::get<N>(m_objectives);
	}
	
private:
	template<int ... I>
	void setupRotations(detail::integer_sequence<I...>){
		m_rotations.insert(m_rotations.begin(), {RotatedObjectiveFunction(&std::get<I>(m_objectives))... });
	}
	std::tuple<Objectives...> m_objectives;
	std::vector<RotatedObjectiveFunction> m_rotations;
};


}
#endif
