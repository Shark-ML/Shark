//===========================================================================
/*!
 * 
 * \file        DataDistribution.h
 *
 * \brief       Learning problems given by analytic distributions.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2006-2013
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
//===========================================================================


#ifndef SHARK_DATA_DATADISTRIBUTION_H
#define SHARK_DATA_DATADISTRIBUTION_H

#include <shark/Data/Dataset.h>
#include <shark/Rng/GlobalRng.h>
#include <utility>

namespace shark {


///
/// \brief A DataDistribution defines an unsupervised learning problem.
///
/// \par
/// The unsupervised learning problem is defined by an explicit
/// distribution (in contrast to a finite dataset). The only
/// method we need is to draw a sample from the distribution.
///
template <class InputType>
class DataDistribution
{
public:
	/// \brief Virtual destructor.
	virtual ~DataDistribution() { }

	/// \brief Generates a single pair of input and label.
	///
	/// @param input the generated input
	virtual void draw(InputType& input) const = 0;

	// \brief Interface for std::generate.
	InputType operator() () {
		InputType ret;
		draw(ret);
		return ret;
	}
	
	/// \brief Generates a data set with samples from from the distribution.
	///
	/// @param size the number of samples in the dataset
	/// @param maximumBatchSize the maximum size of a batch
	UnlabeledData<InputType> generateDataset(std::size_t size,std::size_t maximumBatchSize) const {
		std::size_t batches = (size + maximumBatchSize - 1) / maximumBatchSize;
		std::size_t optimalBatchSize = size / batches;
		std::size_t remainder = size - batches * optimalBatchSize;
		UnlabeledData<InputType> dataset(batches);
		InputType input;

		// now create and fill the batches, taking the remainder into account
		for (std::size_t i=0; i<batches; ++i)
		{
			std::size_t batchsize = (i<remainder) ? optimalBatchSize + 1 : optimalBatchSize;
			typename UnlabeledData<InputType>::batch_reference b = dataset.batch(i);
			draw(input);
			b = Batch<InputType>::createBatch(input, batchsize);
			for (std::size_t j=0; j<batchsize; j++)
			{
				if (j != 0) draw(input);
				shark::get(b, j) = input;
			}
		}
		return dataset;
	}
	
	/// \brief Generates a data set with samples from from the distribution.
	///
	/// @param size the number of samples in the dataset
	UnlabeledData<InputType> generateDataset(std::size_t size) const {
		return generateDataset(size,Data<InputType>::DefaultBatchSize );
	}
};


///
/// \brief A LabeledDataDistribution defines a supervised learning problem.
///
/// \par
/// The supervised learning problem is defined by an explicit
/// distribution (in contrast to a finite dataset). The only
/// method we need is to draw a sample from the distribution.
///
template <class InputType, class LabelType>
class LabeledDataDistribution
{
public:
	/// \brief Virtual destructor.
	virtual ~LabeledDataDistribution() { }

	/// \brief Generates a single pair of input and label.
	/// @param input the generated input
	/// @param label the generated label
	virtual void draw(InputType& input, LabelType& label) const = 0;

	// \Brief Interface for std::generate.
	std::pair<InputType,LabelType> operator() () {
		std::pair<InputType,LabelType> ret;
		draw(ret.first,ret.second);
		return ret;
	}
	
	/// \brief Generates a dataset with samples from from the distribution.
	///
	/// @param size the number of samples in the dataset
	/// @param maximumBatchSize the maximum size of a batch
	LabeledData<InputType, LabelType> generateDataset(std::size_t size,std::size_t maximumBatchSize) const
	{
		// first determine the optimal number of batches and their sizes
		std::size_t batches = (size + maximumBatchSize - 1) / maximumBatchSize;
		std::size_t optimalBatchSize = size / batches;
		std::size_t remainder = size - batches * optimalBatchSize;
		LabeledData<InputType, LabelType> dataset(batches);
		InputType input;
		LabelType label;
		DataPair<InputType, LabelType> pair(input, label);

		// now create and fill the batches, taking the remainder into account
		for (std::size_t i=0; i<batches; ++i)
		{
			std::size_t batchsize = (i<remainder) ? optimalBatchSize + 1 : optimalBatchSize;
			typename LabeledData<InputType, LabelType>::batch_reference b = dataset.batch(i);
			draw(input, label); pair.input = input; pair.label = label;
			b = Batch<DataPair<InputType, LabelType> >::createBatch(pair, batchsize);
			for (std::size_t j=0; j<batchsize; j++)
			{
				if (j != 0) draw(input, label);
				shark::get(b, j).input = input;
				shark::get(b, j).label = label;
			}
		}
		return dataset;
	}
	
	/// \brief Generates a data set with samples from from the distribution.
	///
	/// @param size the number of samples in the dataset
	LabeledData<InputType, LabelType> generateDataset(std::size_t size) const {
		return generateDataset(size,LabeledData<InputType, LabelType>::DefaultBatchSize );
	}
};


///
/// \brief "chess board" problem for binary classification
///
class Chessboard : public LabeledDataDistribution<RealVector, unsigned int>
{
public:
	Chessboard(unsigned int size = 4, double noiselevel = 0.0)
	{
		m_size = size;
		m_noiselevel = noiselevel;
	}


	void draw(RealVector& input, unsigned int& label)const{
		input.resize(2);
		unsigned int j, t = 0;
		for (j = 0; j < 2; j++)
		{
			double v = Rng::uni(0.0, (double)m_size);
			t += (int)floor(v);
			input(j) = v;
		}
		label = (t & 1);
		if (Rng::uni(0.0, 1.0) < m_noiselevel) label = 1 - label;
	}

protected:
	unsigned int m_size;
	double m_noiselevel;
};


///
/// \brief Noisy sinc function: y = sin(x) / x + noise
///
class Wave : public LabeledDataDistribution<RealVector, RealVector>
{
public:
	Wave(double stddev = 0.1, double range = 5.0){
		m_stddev = stddev;
		m_range = range;
	}


	void draw(RealVector& input, RealVector& label)const{
		input.resize(1);
		label.resize(1);
		input(0) = Rng::uni(-m_range, m_range);
		if(input(0) != 0)
            label(0) = sin(input(0)) / input(0) + Rng::gauss(0.0, m_stddev);
        else
            label(0) = Rng::gauss(0.0, m_stddev);
	}

protected:
	double m_stddev;
	double m_range;
};



/// "Pami Toy" problem for binary classification, as used in the article "Glasmachers
/// and C. Igel. Maximum Likelihood Model Selection for 1-Norm Soft Margin SVMs with Multiple 
/// Parameters. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2010."
/// In summary, the first M dimensions are correlated to the labels, the last N dimensions
/// are not. 
class PamiToy : public LabeledDataDistribution<RealVector, unsigned int>
{
public:
	PamiToy(unsigned int size_useful = 5, unsigned int size_noise = 5, double noise_position = 0.0, double noise_variance = 1.0 )
	: m_size( size_useful+size_noise ),
	  m_sizeUseful( size_useful ),
	  m_sizeNoise( size_noise ),
	  m_noisePos( noise_position) ,
	  m_noiseVar( noise_variance )
	{ }

	void draw(RealVector& input, unsigned int& label)const{
		input.resize( m_size );
		label = Rng::discrete( 0, 1 ); //fix label first
		double y2 = label - 0.5; //"clean" informative feature values
		// now fill the informative features..
		for ( unsigned int i=0; i<m_sizeUseful; i++ ) {
			input(i) = y2 + Rng::gauss( m_noisePos, m_noiseVar );
		}
		// ..and the uninformative ones
		for ( unsigned int i=m_sizeUseful; i<m_size; i++ ) {
			input(i) = Rng::gauss( m_noisePos, m_noiseVar );
		}
	}

protected:
	unsigned int m_size;
	unsigned int m_sizeUseful;
	unsigned int m_sizeNoise;
	double m_noisePos;
	double m_noiseVar;
};

/// This class randomly fills a (hyper-)square with data points. Points which 
/// happen to be within a (hyper-)circle centered in the square of a certain
/// radius get a positive class label. Noise on the labels can be added.
class CircleInSquare : public LabeledDataDistribution<RealVector, unsigned int>
{
public:
	CircleInSquare( unsigned int dimensions = 2, double noiselevel = 0.0, bool class_prob_equal = false )
	: m_dimensions( dimensions ),
	  m_noiselevel( noiselevel ),
	  m_lowerLimit( -1 ),
	  m_upperLimit( 1 ),
	  m_centerpoint( 0 ),
	  m_inner_radius2( 0.5*0.5 ),
	  m_outer_radius2( 0.5*0.5 ),
	  m_equal_class_prob( class_prob_equal )
	{ }
	
	/// allow for arbitrary box limits
	void setLimits( double lower_limit, double upper_limit, double inner_radius, double outer_radius )
	{
		RANGE_CHECK( lower_limit < upper_limit );
		RANGE_CHECK( inner_radius <= outer_radius );
		RANGE_CHECK( 2*outer_radius <= upper_limit-lower_limit );
		m_lowerLimit = lower_limit;
		m_upperLimit = upper_limit;
		m_centerpoint = (upper_limit-lower_limit)/2.0;
		m_inner_radius2 = inner_radius*inner_radius;
		m_outer_radius2 = outer_radius*outer_radius;
	}
	
	void draw(RealVector& input, unsigned int& label)const
	{
		input.resize( m_dimensions );
		double v, dist;
		
		if ( m_equal_class_prob ) { //each class has equal probability - this implementation is brute-force and gorgeously inefficient :/
			bool this_label = Rng::coinToss();
			label = ( this_label ? 1 : 0 );
			if ( Rng::uni(0.0, 1.0) < m_noiselevel )
				label = 1 - label;
			if ( this_label ) {
				do {
					dist = 0.0;
					for ( unsigned int i=0; i<m_dimensions; i++ ) {
						v = Rng::uni( m_lowerLimit, m_upperLimit );
						input(i) = v;
						dist += (v-m_centerpoint)*(v-m_centerpoint);
					}
				} while( dist > m_inner_radius2 );
			}
			else {
				do {
					dist = 0.0;
					for ( unsigned int i=0; i<m_dimensions; i++ ) {
						v = Rng::uni( m_lowerLimit, m_upperLimit );
						input(i) = v;
						dist += (v-m_centerpoint)*(v-m_centerpoint);
					}
				} while( dist < m_outer_radius2 );
			}
		}
		else { //equal probability to be anywhere in the cube
			do {
				dist = 0.0;
				for ( unsigned int i=0; i<m_dimensions; i++ ) {
					v = Rng::uni( m_lowerLimit, m_upperLimit );
					input(i) = v;
					dist += (v-m_centerpoint)*(v-m_centerpoint);
				}
				label = ( dist < m_inner_radius2 ? 1 : 0 );
				if ( Rng::uni(0.0, 1.0) < m_noiselevel )
					label = 1 - label;
			} while( dist > m_inner_radius2 && dist < m_outer_radius2 );
		}
	}

protected:
	unsigned int m_dimensions;
	double m_noiselevel;
	double m_lowerLimit;
	double m_upperLimit;
	double m_centerpoint;
	double m_inner_radius2;
	double m_outer_radius2;
	bool m_equal_class_prob; ///<if true, the probability to belong to either class is equal. if false, it is uniform over the cube.
};

// This class randomly fills a 4x4 square in the 2D-plane with data points. 
// Points in the lower left diagonal half are negative, points in the
// upper right diagonal half are positive. But additionally, all points
// in a circle located in the lower right quadrant are positive, effectively
// bulging the decision boundary inward. Noise on the labels can be added.
class DiagonalWithCircle : public LabeledDataDistribution<RealVector, unsigned int>
{
public:
	DiagonalWithCircle( double radius = 1.0, double noise = 0.0 )
	: m_radius2( radius*radius ),
	  m_noiselevel( noise )
	{ }
	
	void draw(RealVector& input, unsigned int& label)const
	{
		input.resize( 2 );
		double x,y;
		x = Rng::uni( 0, 4 ); //zero is left
		y = Rng::uni( 0, 4 ); //zero is bottom
		// assign label according to position w.r.t. the diagonal
		if ( x+y < 4 )
			label = 1;
		else
			label = 0;
		// but if in the circle (even above diagonal), assign positive label
		if ( (3-x)*(3-x) + (1-y)*(1-y) < m_radius2 )
			label = 1;
		
		// add noise
		if ( Rng::uni(0.0, 1.0) < m_noiselevel )
			label = 1 - label;
		input(0) = x;
		input(1) = y;
	}

protected:
	double m_radius2;
	double m_noiselevel;
};

}
#endif
