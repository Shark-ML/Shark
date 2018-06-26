//===========================================================================
/*!
 * 
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


#ifndef SHARK_DATA_DATADISTRIBUTION_H
#define SHARK_DATA_DATADISTRIBUTION_H

#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/Core/Random.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
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
	typedef Data<InputType> DatasetType;
	typedef typename DatasetType::shape_type shape_type;

	DataDistribution(shape_type const& shape):m_shape(shape){}

	/// \brief Virtual destructor.
	virtual ~DataDistribution() { }

	/// \brief Generates a single pair of input and label.
	///
	/// @param input the generated input
	virtual void draw(InputType& input) const = 0;
	
	/// \brief Returns the shape of the dataset.
	shape_type const& shape() const{
		return m_shape;
	}
	
	/// \brief Generates a data set with samples from from the distribution.
	///
	/// @param size the number of samples in the dataset
	/// @param maximumBatchSize the maximum size of a batch
	Data<InputType> generateDataset(std::size_t size,std::size_t maximumBatchSize = constants::DefaultBatchSize) const {
		DatasetType data(size, shape(), maximumBatchSize);

		// draw the samples
		InputType input;
		for(auto&& element: data.elements()){
			draw(input);
			element = input;
		}
		return data;
	}
private:
	shape_type m_shape;
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
	typedef LabeledData<InputType, LabelType> DatasetType;
	typedef typename DatasetType::shape_type shape_type;

	LabeledDataDistribution(shape_type const& shape):m_shape(shape){}

	/// \brief Generates a single pair of input and label.
	/// @param input the generated input
	/// @param label the generated label
	virtual void draw(InputType& input, LabelType& label) const = 0;
	
	/// \brief Returns the shape of the dataset.
	shape_type const& shape() const{
		return m_shape;
	}
	
	/// \brief Generates a dataset with samples from from the distribution.
	///
	/// @param size the number of samples in the dataset
	/// @param maximumBatchSize the maximum size of a batch
	DatasetType generateDataset(std::size_t size,std::size_t maximumBatchSize = constants::DefaultBatchSize) const{
		DatasetType data(size, shape(), maximumBatchSize);

		// draw the samples
		InputLabelPair<InputType,LabelType> pair;
		for(auto&& element: data.elements()){
			draw(pair.input,pair.label);
			element = pair;
		}
		return data;
	}
private:
	shape_type m_shape;
};


///
/// \brief "chess board" problem for binary classification
///
class Chessboard : public LabeledDataDistribution<RealVector, unsigned int>
{
public:
	Chessboard(unsigned int size = 4, double noiselevel = 0.0)
	:LabeledDataDistribution<RealVector, unsigned int>({2,2}){
		m_size = size;
		m_noiselevel = noiselevel;
	}


	void draw(RealVector& input, unsigned int& label)const{
		input.resize(2);
		unsigned int j, t = 0;
		for (j = 0; j < 2; j++)
		{
			double v = random::uni(random::globalRng, 0.0, (double)m_size);
			t += (int)floor(v);
			input(j) = v;
		}
		label = (t & 1);
		if (random::uni(random::globalRng, 0.0, 1.0) < m_noiselevel) label = 1 - label;
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
	Wave(double stddev = 0.1, double range = 5.0)
	: LabeledDataDistribution<RealVector, RealVector>({1,1}){
		m_stddev = stddev;
		m_range = range;
	}

	void draw(RealVector& input, RealVector& label)const{
		input.resize(1);
		label.resize(1);
		input(0) = random::uni(random::globalRng, -m_range, m_range);
		if(input(0) != 0)
            label(0) = sin(input(0)) / input(0) + random::gauss(random::globalRng, 0.0, m_stddev);
        else
            label(0) = random::gauss(random::globalRng, 0.0, m_stddev);
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
	: LabeledDataDistribution<RealVector, unsigned int>({size_useful+size_noise,2})
	, m_size( size_useful+size_noise )
	, m_sizeUseful( size_useful )
	, m_sizeNoise( size_noise )
	, m_noisePos( noise_position)
	, m_noiseVar( noise_variance )
	{ }

	void draw(RealVector& input, unsigned int& label)const{
		input.resize( m_size );
		label =  (unsigned int) random::discrete(random::globalRng, 0,1); //fix label first
		double y2 = label - 0.5; //"clean" informative feature values
		// now fill the informative features..
		for ( unsigned int i=0; i<m_sizeUseful; i++ ) {
			input(i) = y2 + random::gauss(random::globalRng, m_noisePos, m_noiseVar );
		}
		// ..and the uninformative ones
		for ( unsigned int i=m_sizeUseful; i<m_size; i++ ) {
			input(i) = random::gauss(random::globalRng, m_noisePos, m_noiseVar );
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
	: LabeledDataDistribution<RealVector, unsigned int>({2, 2})
	, m_dimensions( dimensions )
	, m_noiselevel( noiselevel )
	, m_lowerLimit( -1 )
	, m_upperLimit( 1 )
	, m_centerpoint( 0 )
	, m_inner_radius2( 0.5*0.5 )
	, m_outer_radius2( 0.5*0.5 )
	, m_equal_class_prob( class_prob_equal )
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
			bool this_label = random::coinToss(random::globalRng);
			label = ( this_label ? 1 : 0 );
			if ( random::uni(random::globalRng, 0.0, 1.0) < m_noiselevel )
				label = 1 - label;
			if ( this_label ) {
				do {
					dist = 0.0;
					for ( unsigned int i=0; i<m_dimensions; i++ ) {
						v = random::uni(random::globalRng, m_lowerLimit, m_upperLimit );
						input(i) = v;
						dist += (v-m_centerpoint)*(v-m_centerpoint);
					}
				} while( dist > m_inner_radius2 );
			}
			else {
				do {
					dist = 0.0;
					for ( unsigned int i=0; i<m_dimensions; i++ ) {
						v = random::uni(random::globalRng, m_lowerLimit, m_upperLimit );
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
					v = random::uni(random::globalRng, m_lowerLimit, m_upperLimit );
					input(i) = v;
					dist += (v-m_centerpoint)*(v-m_centerpoint);
				}
				label = ( dist < m_inner_radius2 ? 1 : 0 );
				if ( random::uni(random::globalRng, 0.0, 1.0) < m_noiselevel )
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
	: LabeledDataDistribution<RealVector, unsigned int>({2, 2})
	, m_radius2( radius*radius )
	, m_noiselevel( noise )
	{ }
	
	void draw(RealVector& input, unsigned int& label)const{
		input.resize( 2 );
		double x,y;
		x = random::uni(random::globalRng, 0, 4 ); //zero is left
		y = random::uni(random::globalRng, 0, 4 ); //zero is bottom
		// assign label according to position w.r.t. the diagonal
		if ( x+y < 4 )
			label = 1;
		else
			label = 0;
		// but if in the circle (even above diagonal), assign positive label
		if ( (3-x)*(3-x) + (1-y)*(1-y) < m_radius2 )
			label = 1;
		
		// add noise
		if ( random::uni(random::globalRng, 0.0, 1.0) < m_noiselevel )
			label = 1 - label;
		input(0) = x;
		input(1) = y;
	}

protected:
	double m_radius2;
	double m_noiselevel;
};


/// \brief Generates a set of normally distributed points
class NormalDistributedPoints:public DataDistribution<RealVector>
{
public:
	/// \brief Generates a simple distribution with 
	NormalDistributedPoints(std::size_t dim)
	: DataDistribution<RealVector>(dim)
	, m_offset(dim,0){
		RealMatrix covariance(dim,dim,0);
		diag(covariance) = blas::repeat(1.0,dim);
		m_dist.setCovarianceMatrix(covariance);
	}
	NormalDistributedPoints(RealMatrix const& covariance, RealVector const& offset)
	: DataDistribution<RealVector>(offset.size())
	, m_dist(covariance), m_offset(offset){
		SIZE_CHECK(offset.size() == covariance.size1());
	}
	void draw(RealVector& input) const{
		input.resize(m_offset.size());
		noalias(input) = m_offset;
		noalias(input) += m_dist(random::globalRng).first;
	}
	
private:
	MultiVariateNormalDistributionCholesky m_dist;
	RealVector m_offset;
};

/// \brief Given a set of images, draws a set of image patches of a given size
class ImagePatches:public DataDistribution<RealVector>{
public:
	ImagePatches(
		Data<RealVector> images, 
		std::size_t imageWidth, std::size_t imageHeight,
		std::size_t patchWidth, std::size_t patchHeight
	):DataDistribution<RealVector>({m_patchWidth, m_patchHeight})
	, m_images(images)
	, m_imageWidth(imageWidth)
	, m_imageHeight(imageHeight)
	, m_patchWidth(patchWidth)
	, m_patchHeight(patchHeight){}
		
	void draw(RealVector& input) const{
		//sample image
		std::size_t imageNum = random::discrete(random::globalRng, std::size_t(0),m_images.size()-1);
		auto image = m_images[imageNum];
		//draw the upper left corner of the image
		std::size_t m_startX = random::discrete(random::globalRng, std::size_t(0),m_imageWidth-m_patchWidth);
		std::size_t m_startY = random::discrete(random::globalRng, std::size_t(0),m_imageHeight-m_patchHeight);
		
		
		//copy patch
		input.resize(m_patchWidth * m_patchHeight);
		std::size_t rowStart = m_startY * m_imageWidth + m_startX;
		for (size_t y = 0; y < m_patchHeight; ++y){
			for (size_t x = 0; x < m_patchWidth; ++x){
				input(y * m_patchWidth + x) = image(rowStart+x);
			}
			rowStart += m_imageWidth;
		}
	}
private:
	DataView<Data<RealVector> > m_images;
	std::size_t m_imageWidth;
	std::size_t m_imageHeight;
	std::size_t m_patchWidth;
	std::size_t m_patchHeight;
};

}
#endif
