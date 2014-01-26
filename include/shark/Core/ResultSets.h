/*!
 * 
 *
 * \brief       Result sets for algorithms.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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

#ifndef SHARK_CORE_RESULTSETS_H
#define SHARK_CORE_RESULTSETS_H

#include <ostream>

namespace shark{
	
template<class SearchPointT, class ResultT>
struct ResultSet{
	typedef SearchPointT SearchPointType;
	typedef ResultT ResultType;
	ResultSet():value(/*null*/){}
	ResultSet(ResultType const& value, SearchPointType const& point)
	:point(point),value(value){}

	SearchPointType point;
	ResultType value;
	
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int /*version*/ ) {
		archive & point;
		archive & value;
	}
};

/// \brief Generates a typed solution given the search point and the corresponding objective function value.
///
/// \param [in] t The search point.
/// \param [in] u The objective function value.
///
/// \returns A ResultSet containing the supplied search point and objective function value.
template<typename T, typename U>
ResultSet<T,U> makeResultSet(T const& t, U const& u ) {
	return ResultSet<T,U>( u, t );
}
template<class SearchPoint,class Result>
std::ostream & operator<<( std::ostream & out, ResultSet<SearchPoint,Result> const& solution  ) {
	out << solution.value << " " << solution.point;
	return out;
}

///\brief Result set for single objective algorithm.
///
///Contains a point of the search space as well its value on the objective function.
template<class SearchPointTypeT>
struct SingleObjectiveResultSet: public ResultSet<SearchPointTypeT,double>{
	typedef SearchPointTypeT SearchPointType;
	typedef double ResultType;
	
	SingleObjectiveResultSet(){}
	SingleObjectiveResultSet(double value, SearchPointType const& point)
	:ResultSet<SearchPointTypeT,double>(value, point){}
		
	///\brief Compares two SingleObjectiveResultSets. Returns true if op1.value < op2.value.
	friend bool operator<(SingleObjectiveResultSet const& op1, SingleObjectiveResultSet const& op2){
		return op1.value < op2.value;
	}
};




///\brief Result set for validated points.
///
///If validation is applied, this error function additionally saves the value on the validation set.
///order between sets is by the validation error.
template<class SearchPointTypeT>
struct ValidatedSingleObjectiveResultSet:public SingleObjectiveResultSet<SearchPointTypeT>  {
private:
	typedef SingleObjectiveResultSet<SearchPointTypeT> base_type;
public:
	ValidatedSingleObjectiveResultSet():validation(0){}
	ValidatedSingleObjectiveResultSet(base_type const& base)
	:base_type(base),validation(0){}
	ValidatedSingleObjectiveResultSet(base_type const& base, double validation)
	:base_type(base),validation(validation){}

	typename base_type::ResultType validation;
		
	template<typename Archive>
	void serialize( Archive & archive, const unsigned int /*version*/ ) {
		archive & boost::serialization::base_object<base_type >(*this);
		archive & validation;
	}
	
	/// \brief Compares two ValidatedSingleObjectiveResultSets. Returns true if op1.validation < op2.validation
	friend bool operator<(ValidatedSingleObjectiveResultSet const& op1, ValidatedSingleObjectiveResultSet const& op2){
		return op1.validation < op2.validation;
	}
};

template<class SearchPoint>
std::ostream & operator<<( std::ostream & out, ValidatedSingleObjectiveResultSet<SearchPoint> const& solution  ) {
	out << solution.validation << " "<< solution.value << " " << solution.point;
	return out;
}
}

#endif
