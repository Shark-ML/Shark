//===========================================================================
/*!
 *  \file LinearClassifier.cpp
 *
 *  \brief LinearClassifier
 *
 *  \author O.Krause
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#include <shark/Models/LinearClassifier.h>
#include <shark/LinAlg/solveSystem.h>
#include <shark/LinAlg/BLAS/Initialize.h>
#include <limits>

using namespace shark;

void LinearClassifier::setStructure(std::size_t inputSize,std::size_t classes){
	m_inverseCholesky.resize(inputSize,inputSize);
	m_means.resize(classes,inputSize);
	m_transformedMeans.resize(classes,inputSize);
	m_classBias.resize(classes);
}

void LinearClassifier::configure( PropertyTree const& node ){
	std::size_t inputs=node.get<std::size_t>("inputs");
	std::size_t classes=node.get<unsigned int>("classes");
	setStructure(inputs,classes);
}

RealVector LinearClassifier::parameterVector()const{
	RealVector parameters(numberOfParameters());
	init(parameters)<<toVector(m_inverseCholesky), toVector(m_means);
	return parameters;
}
void LinearClassifier::setParameterVector(RealVector const& parameters){
	init(parameters) >> toVector(m_inverseCholesky), toVector(m_means);
	//update transformation
	fast_prod(m_means,m_inverseCholesky,m_transformedMeans);
	for(std::size_t classID = 0; classID != numberOfClasses(); ++classID){
		RealMatrixRow mean=row(m_transformedMeans,classID);
		m_classBias(classID)=normSqr(mean);
	}
}

void LinearClassifier::setClassMean(std::size_t classID, RealVector const& mean){
	SIZE_CHECK(classID<numberOfClasses());
	noalias(row(m_means,classID)) = mean;
	fast_prod(trans(m_inverseCholesky),row(m_means,classID),row(m_transformedMeans,classID));
	m_classBias(classID)=normSqr(row(m_transformedMeans,classID));
}

std::size_t LinearClassifier::numberOfParameters()const{
	std::size_t input=inputSize();
	std::size_t classes=numberOfClasses();
	return input*input+classes*input;
}
void LinearClassifier::importCovarianceMatrix(RealMatrix const& covariance){
	//compute inverse cholesky decomposition from the covariance matrix
	//to prevent problems due to noninvertible covariance matrices, we use
	//a general inverse decomposition here which calculates U with C'=UU^T
	//and C' being the pseudo inverse hessian
	decomposedGeneralInverse(covariance,m_inverseCholesky);
//	//first cholesky decomposition C=AA^T
//	RealMatrix cholesky(inputSize(),inputSize());
//	choleskyDecomposition(covariance,cholesky);
//	//now solve the lower triangular system A*B=I with B being the result and I the identity
//	noalias(m_inverseCholesky)=RealIdentityMatrix(inputSize());
//	solveTriangularSystemInPlace<SolveAXB,blas::lower_tag>(cholesky,m_inverseCholesky);
	
	//update transformation
	fast_prod(m_means,m_inverseCholesky,m_transformedMeans);
	for(std::size_t classID = 0; classID != numberOfClasses(); ++classID){
		RealMatrixRow mean=row(m_transformedMeans,classID);
		m_classBias(classID)=normSqr(mean);
	}
}


void LinearClassifier::eval(RealMatrix const& patterns,UIntVector& output)const{
	SIZE_CHECK(patterns.size2()==inputSize());
	
	//we calculate the class means using the formula (<,> is inner prod)
	//< a-c,C(a-c) >
	//where c is the class mean, "a" a single pattern and C the inverse covariance
	//using cholesky decomposition C=AA^T and a'=A^Ta c'=A^Tc we can write this also as:
	//< a-c,C(a-c) > = < a-c,AA^T(a-c) > =  < A^T(a-c),A^T(a-c) > = <a'-c',a'-c'>= <a',a'>-2<c',a'>+<c',c'>
	//we have allready precomputed c' and <c',c'>(the class bias). so we need to compute only a'
	RealMatrix transA(patterns.size1(),inputSize());
	fast_prod(patterns,m_inverseCholesky,transA);
	
	output.resize(patterns.size1());

	for(std::size_t i = 0; i != patterns.size1();++i){
		RealMatrixRow a=row(transA,i);
		double aSquared=normSqr(a);
		double bestDistance = std::numeric_limits<double>::max();
		for (std::size_t c=0; c != numberOfClasses(); c++){
			double distance = aSquared-2*inner_prod(row(m_transformedMeans,c),a);
			distance += m_classBias(c);
			if (distance < bestDistance){
				bestDistance = distance;
				output(i) = c;
			}
		}
	}
}

/// From ISerializable, reads a model from an archive
void LinearClassifier::read( InArchive & archive ){
	archive >> m_inverseCholesky;
	archive >> m_means;
}

/// From ISerializable, writes a model to an archive
void LinearClassifier::write( OutArchive & archive ) const{
	archive << m_inverseCholesky;
	archive << m_means;
}


Data<RealVector> LinearClassifier::softMembership(Data<InputType> const & patterns) const {
		Data<RealVector> prediction(patterns.numberOfBatches());
		for(std::size_t i = 0; i != patterns.numberOfBatches();++i){
			softMembership(patterns.batch(i),prediction.batch(i));
		}
		return prediction;
}

void LinearClassifier::softMembership(BatchInputType const& patterns, Batch<RealVector>::type& output)const{
	//we calculate the class means using the formula (<,> is inner prod)
	//< a-c,C(a-c) >
	//where c is the class mean, "a" a single pattern and C the inverse covariance
	//using cholesky decomposition C=AA^T and a'=A^Ta c'=A^Tc we can write this also as:
	//< a-c,C(a-c) > = < a-c,AA^T(a-c) > =  < A^T(a-c),A^T(a-c) > = <a'-c',a'-c'>= <a',a'>-2<c',a'>+<c',c'>
	//we have allready precomputed c' and <c',c'>(the class bias). so we need to compute only a'
	RealMatrix transA(patterns.size1(),inputSize());
	fast_prod(patterns,m_inverseCholesky,transA);
		
	output.resize(patterns.size1(), numberOfClasses());
	
	for(std::size_t i = 0; i != patterns.size1();++i){
		RealMatrixRow a=row(transA,i);
		double aSquared=normSqr(a);
		double sum = 0;
		
		for (std::size_t c=0; c != numberOfClasses(); c++) {
			double d = aSquared-2*inner_prod(row(m_transformedMeans,c),a) + m_classBias(c);
			d = std::exp(-d);
			sum += d;
			output(i, c) = d;
		}
		for (std::size_t c=0; c != numberOfClasses(); c++) {
			output(i, c) = output(i, c) / sum;
		}
	}
} 
