/*!
 * 
 *
 * \brief       Calculate statistics given a range of values.
 * 
 * 
 *
 * \author     O.Krause
 * \date        2015
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <https://shark-ml.github.io/Shark/>
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
#ifndef SHARK_STATISTICS_H
#define SHARK_STATISTICS_H


//for vector algebra
#include <shark/LinAlg/Base.h>

//handling of missing values
#include <limits>
#include <boost/math/special_functions/fpclassify.hpp>

//for quantiles
#include <boost/range/algorithm/nth_element.hpp>


//for the result table
#include <string>
#include <map>
#include <iterator>
#include <boost/serialization/string.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

namespace shark {
namespace statistics{
	
inline double missingValue(){
	return std::numeric_limits<double>::quiet_NaN();//missing values are a non-signaling NaN
}

inline bool isMissing(double value){
	return boost::math::isnan(value);//there is no portable way to distinguish the different types of NaN
}
	
///\brief Base class for all Statistic Objects to be used with Statistics	
class BaseStatisticsObject{
public:
	virtual std::string name() const=0;
	virtual ~BaseStatisticsObject(){}
	virtual RealVector statistics(std::vector<RealVector> const& points)const=0;
};

///\brief for a vector of points computes for every dimension the fraction of missing values
class FractionMissing:public BaseStatisticsObject{
public:
	std::string name() const{
		return "Missing";
	}
	RealVector statistics(std::vector<RealVector> const& points)const{
		std::size_t N = points.size();
		RealVector missing(points[0].size(),0.0);
		for(std::size_t i  = 0; i != N;++i){
			for(std::size_t j = 0; j != missing.size(); ++j){
				if(!isMissing(points[i](j)))continue;
				missing(j) += 1.0;
			}
		}
		missing /= N;
		return missing;
	}
};

///\brief For a vector of points computes for every dimension the mean
class Mean:public BaseStatisticsObject{
public:
	std::string name() const{
		return "Mean";
	}
	RealVector statistics(std::vector<RealVector> const& points)const{
		std::size_t N = points.size();
		RealVector sum(points[0].size(),0.0);
		UIntVector numSamples(points[0].size(),0);
		for(std::size_t i  = 0; i != N;++i){
			for(std::size_t j = 0; j != sum.size(); ++j){
				if(isMissing(points[i](j)))continue;
				sum(j) += points[i](j);
				++numSamples(j);
			}
		}
		//calculate mean. if the number of non-missing points was 0, return missingValue() for that dimension
		return safe_div(sum,numSamples,missingValue());
	}
};

///\brief For a vector of points computes for every dimension the variance
class Variance:public BaseStatisticsObject{
public:
	std::string name() const{
		return "Variance";
	}
	RealVector statistics(std::vector<RealVector> const& points)const{
		std::size_t N = points.size();
		Mean m;
		RealVector mean = m.statistics(points);
		RealVector variance(mean.size(),0.0);
		UIntVector numSamples(points[0].size(),0);
		for(std::size_t i  = 0; i != N;++i){
			for(std::size_t j = 0; j != mean.size(); ++j){
				if(isMissing(points[i](j)))continue;
				variance(j) += sqr(points[i](j)-mean(j));
				++numSamples(j);
			}
		}
		//calculate biased variance. if the number of non-missing points was 0, return missingValue() for that dimension
		return safe_div(variance,numSamples,missingValue());
	}
};

//Quantiles, Median, Lower-Upper
///\brief For a vector of points computes for every dimension the p-quantile
class Quantile:public BaseStatisticsObject{
public:
	std::string name() const{
		return boost::lexical_cast<std::string>(m_quantile)+"-Quantile";
	}
	Quantile(double quantile):m_quantile(quantile){}
	RealVector statistics(std::vector<RealVector> const& points)const{
		std::size_t N = points.size();
		RealVector quantiles(points[0].size(),missingValue());
		for(std::size_t j = 0; j != quantiles.size(); ++j){
			//get all non-missing values of the j-th dimension
			std::vector<double> values;
			for(std::size_t i  = 0; i != N;++i){
				if(isMissing(points[i](j)))continue;
				values.push_back(points[i](j));
			}
			if(values.size() == 0) continue;//no values-> missing value
			
			//compute quantile of j-th dimension
			std::size_t element = std::size_t(values.size()*m_quantile);
			std::vector<double>::iterator pos= values.begin()+element;
			boost::nth_element(values,pos);
			quantiles(j) = *pos;
		}
		return quantiles;
	}
private:
	double m_quantile;
};

///\brief For a vector of points computes for every dimension the median
class Median:public Quantile{
public:
	std::string name() const{
		return "Median";
	}
	Median():Quantile(0.5){}
};

///\brief For a vector of points computes for every dimension the 25%-quantile
class LowerQuantile:public Quantile{
public:
	LowerQuantile():Quantile(0.25){}
};
///\brief For a vector of points computes for every dimension the 75%-quantile
class UpperQuantile:public Quantile{
public:
	UpperQuantile():Quantile(0.75){}
};


///\brief Stores results of a running experiment
///
/// This is a simple three dimensional table with the dimensions. Experiments
/// are thought of having a varied parameter (for example the algorithm names when
/// several algorithms are compared) and for each parameter a set of vector valued points
/// is stored - one vector for each trial of the experiment for a given parameter.
/// It is posible to give every parameter and the whole table a name which adds meta
/// information, for example to generate outputs.
template<class Parameter>
class ResultTable{
public:
	typedef typename std::map<Parameter, std::vector<RealVector> >::const_iterator const_iterator;

	ResultTable(std::size_t numDimensions, std::string const& parameterName="unnamed")
	:m_dimensionNames(numDimensions,"unnamed"),m_parameterName(parameterName){}

	std::string const& parameterName()const{
		return m_parameterName;
	}		
	
	void setDimensionName(std::size_t i, std::string const& name){
		m_dimensionNames[i]=name;
	}
	
	std::string const& dimensionName(std::size_t i)const{
		return m_dimensionNames[i];
	}
	
	std::size_t numDimensions()const{
		return m_dimensionNames.size();
	}
	
	void update(Parameter const& parameter, RealVector const& point){
		SIZE_CHECK(point.size() == numDimensions());
		m_results[parameter].push_back(point);
	}
	
	void update(Parameter const& parameter, double value){
		RealVector point(1,value);
		update(parameter, point);
	}
	
	void update(Parameter const& parameter, double value1, double value2){
		RealVector point(2);
		point(0)=value1;
		point(1)=value2;
		update(parameter, point);
	}
	
	void update(Parameter const& parameter, double value1, double value2,double value3){
		RealVector point(3);
		point(0)=value1;
		point(1)=value2;
		point(2)=value3;
		update(parameter, point);
	}
	
	std::vector<RealVector>const& operator[](Parameter const& param)const{
		return m_results.find(param)->second;
	}
	
	const_iterator begin()const{
		return m_results.begin();
	}
	const_iterator end()const{
		return m_results.end();
	}
	
	std::size_t numParams()const{
		return m_results.size();
	}
	
	Parameter const& parameterValue(std::size_t i)const{
		const_iterator pos = begin();
		std::advance(pos,i);
		return pos->first;
	}
	
	
	template<class Archive>
	void serialize(Archive &ar, const unsigned int file_version) {
		ar & m_dimensionNames;
		ar & m_parameterName;
		ar & m_results;
		(void) file_version;//prevent warning
	}
	
private:
	std::vector<std::string> m_dimensionNames;
	std::string m_parameterName;
	std::map<Parameter, std::vector<RealVector> > m_results; 
};

///\brief Generates Statistics over the results of an experiment
///
/// Given the results of an experiment stored in a ResultsTable, computes
/// several tatistics for each variable.
template<class Parameter>
struct Statistics {
public:
	typedef typename std::map<Parameter, std::map<std::string,RealVector> >::const_iterator const_iterator;
	Statistics(ResultTable<Parameter> const* table):m_resultsTable(table){}
	
	void addStatistic(std::string const& statisticName, BaseStatisticsObject const& object){
		typedef typename ResultTable<Parameter>::const_iterator iterator;
		iterator end = m_resultsTable->end();
		for(iterator pos=m_resultsTable->begin(); pos != end; ++pos){
			m_statistics[pos->first][statisticName] = object.statistics(pos->second);
		}
		m_statisticNames.push_back(statisticName);
	}
	
	void addStatistic(BaseStatisticsObject const& object){
		addStatistic(object.name(),object);
	}
	
	std::map<std::string,RealVector> const& operator[](Parameter const& parameter)const{
		return m_statistics.find(parameter)->second;
	}
	
	const_iterator begin()const{
		return m_statistics.begin();
	}
	const_iterator end()const{
		return m_statistics.end();
	}
	
	//information about the parameter of the experiments
	std::string const& parameterName()const{
		return m_resultsTable->parameterName();
	}
	
	std::size_t numParams()const{
		return m_resultsTable->numParams();
	}
	
	Parameter const& parameterValue(std::size_t i)const{
		return m_resultsTable->parameterValue(i);
	}
	
	//information about the names of the dimensions
	std::size_t numDimensions()const{
		return m_resultsTable->numDimensions();
	}
	
	std::string const& dimensionName(std::size_t i)const{
		return m_resultsTable->dimensionName(i);
	}
	
	//information about the statistics
	std::size_t numStatistics()const{
		return m_statisticNames.size();
	}
	std::string const& statisticName(std::size_t i)const{
		return m_statisticNames[i];
	}
private:
	std::vector<std::string> m_statisticNames;
	ResultTable<Parameter> const* m_resultsTable;
	std::map<Parameter, std::map<std::string,RealVector> > m_statistics;
};

template<class Parameter>
void printCSV(Statistics<Parameter> const& statistics){
	//first print a legend
	std::cout<<"# "<<statistics.parameterName();
	for(std::size_t i = 0; i != statistics.numStatistics(); ++i){
		for(std::size_t j = 0; j != statistics.numDimensions(); ++j){
			std::cout<<" "<<statistics.statisticName(i)<<"-"<<statistics.dimensionName(j);
		}
	}
	std::cout<<"\n";
	
	//print results parameter by parameter
	for(std::size_t k = 0; k != statistics.numParams(); ++k){
		Parameter param=statistics.parameterValue(k);
		std::map<std::string,RealVector> paramResults=statistics[param];
		std::cout<<param;
		for(std::size_t i = 0; i != statistics.numStatistics(); ++i){
			for(std::size_t j = 0; j != statistics.numDimensions(); ++j){
				std::cout<<" "<<paramResults[statistics.statisticName(i)](j);
			}
		}
		std::cout<<"\n";
	}
}

}}
#endif // SHARK_STATISTICS_H
