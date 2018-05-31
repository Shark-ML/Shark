//===========================================================================
/*!
 * 
 *
 * \author     Oswin Krause
 * \date        2016
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_MULTIOBJECTIVEBENCHMARK_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_MULTIOBJECTIVEBENCHMARK_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/ObjectiveFunctions/Benchmarks/RotatedErrorFunction.h>
#include <shark/ObjectiveFunctions/BoxConstraintHandler.h>
#include <shark/Core/Random.h>

#include <shark/LinAlg/rotations.h>
#include <tuple>

namespace shark {namespace benchmarks{

namespace detail{
	//taken from the web. implements an std::integer_sequence type representing a sequence 0,...,N-1, std::integer_sequence is not here until C++14
	template<int...> struct integer_sequence { using type = integer_sequence; };
	template<typename T1, typename T2> struct integer_sequence_concat;
	template<int... I1, int... I2> struct integer_sequence_concat<integer_sequence<I1...>, integer_sequence<I2...>>: integer_sequence<I1..., (sizeof...(I1) + I2)...> {};

	//generate_integer_sequence generates an integer sequence of integers 0....N. requires log(N) template instantiations
	template<int N> struct generate_integer_sequence;
	template<int N> struct generate_integer_sequence: integer_sequence_concat<typename generate_integer_sequence<N/2>::type, typename generate_integer_sequence<N-N/2>::type>::type {};
	template <> struct generate_integer_sequence<0>: integer_sequence<>{};
	template <> struct generate_integer_sequence<1>: integer_sequence<0>{};
};

/// \brief Creates  a multi-objective Benchmark from a set of given single objective functions
///
/// A variadic template is used to generate a set of benchmarks.
/// eg MultiObjectiveBenchmark<Sphere,Ellispoid,Rosenbrock> sets up a three-objective Benchmark.
///
/// A random rotation and translation is applied to each benchmark function, thus
/// MultiObjectiveBenchmark<Sphere,Sphere> forms a non-degenerate front.
/// the ith objective can be queried via the get<i> member function.
///
/// The generated translations are approximately sampled from the unit ball and starting points are also drawn
/// by the same distribution around a random optimum (assuming the optimum is at (0,0) of the untranslated function
///
/// Note that all objectives must have scalable dimensionality
/// \ingroup benchmarks
template<class ... Objectives>
class MultiObjectiveBenchmark: public MultiObjectiveFunction{
public:
	MultiObjectiveBenchmark(std::size_t numVariables = 5){
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
		setupRotations(typename detail::generate_integer_sequence<sizeof...(Objectives)>::type());
		setNumberOfVariables(numVariables);//prevent that different objectives have different default number of variables.
		for(auto& f: m_rotations){
			//if one function does not have a first derivative, no function has
			if(!f.hasFirstDerivative())
				m_features.reset(HAS_FIRST_DERIVATIVE);
		}
	};
	
	///\brief Name of the Benchmark
	///
	/// The name has the form Objective1/Objective2/Objective3/.../ObjectiveN
	/// where ObjectiveK is the name of the k-th objective.
	std::string name()const{
		return generateName(typename detail::generate_integer_sequence<sizeof...(Objectives)>::type());
		
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}
	
	void setNumberOfVariables( std::size_t numberOfVariables ){
		for(auto& f: m_rotations){
			SHARK_RUNTIME_CHECK(f.hasScalableDimensionality(),"Function is not scalable");
			f.setNumberOfVariables(numberOfVariables);
		}
	}
	
	std::size_t numberOfObjectives()const{
		return sizeof...(Objectives);
	}
	
	std::size_t numberOfVariables()const{
		return get<0>().numberOfVariables();
	}
	
	template<int N>
	typename std::tuple_element<N, std::tuple<Objectives...> >::type& get(){
		return std::get<N>(m_objectives);
	}
	template<int N>
	typename std::tuple_element<N, std::tuple<Objectives...> >::type const& get()const{
		return std::get<N>(m_objectives);
	}
	
	///\ Initializes the functions as well as picks random rotations and translations
	void init() {
		m_translations.clear();
		
		for(auto& f: m_rotations)
		{
			RealVector translation(numberOfVariables());
			for(double& v: translation){
				v=random::gauss(*mep_rng, 0,1)/std::sqrt(double(numberOfVariables()));
			}
			m_translations.push_back(translation);
			f.setRng(mep_rng);
			f.init();
		}
	}
	
	SearchPointType proposeStartingPoint() const {
		RealVector x(numberOfVariables());

		std::size_t index = random::discrete(0,m_rotations.size()-1);
		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = m_translations[index](i)+random::gauss(*mep_rng, 0,1)/std::sqrt(double(numberOfVariables()));
		}
		return x;
	}
	
	/// Returns the vector (f_1(x),...,f_N(x)) of the N objectives in the benchmark for the current point.
	ResultType eval( SearchPointType const& x ) const {
		m_evaluationCounter++;

		ResultType value( numberOfObjectives() );
		for(std::size_t  i = 0; i != value.size(); ++i){
			value(i) = m_rotations[i].eval( x - m_translations[i]);
		}
		return value;
	}
	/// Calculates function value as well as the the Jacobian( d/dxf_1(x),...,d/dx f_N(x)) of the N objectives in the benchmark for the current point.
	ResultType evalDerivative( SearchPointType const& x, FirstOrderDerivative& derivative )const {
		derivative.resize(numberOfObjectives(), numberOfVariables());
		RealVector singleDerivative;
		RealVector value(numberOfObjectives());
		for(std::size_t  i = 0; i != value.size(); ++i){
			value(i) = m_rotations[i].evalDerivative( x - m_translations[i],singleDerivative);
			noalias(row(derivative,i)) = singleDerivative;
		}
		
		return value;
	}
private:
	//generate a rotated objective function for each function in the m_objectives tuple
	template<int ... I>
	void setupRotations(detail::integer_sequence<I...>){
		m_rotations.insert(m_rotations.begin(), {RotatedObjectiveFunction(&std::get<I>(m_objectives))... });
	}
	
	template<int ... I>
	std::string generateName(detail::integer_sequence<I...>)const{
		std::string name;
		for(auto const& fname:{std::get<I>(m_objectives).name()... }){
			name+=fname+'/';
		}
		name.pop_back();
		return name;
	}
	
	std::tuple<Objectives...> m_objectives;
	std::vector<RotatedObjectiveFunction> m_rotations;
	std::vector<RealVector> m_translations;
};


}}
#endif
