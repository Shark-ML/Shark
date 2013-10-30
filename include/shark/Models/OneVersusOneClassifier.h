//===========================================================================
/*!
*
*  \brief One-versus-one Classifier.
*
*  \author  T. Glasmachers
*  \date    2012
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

#ifndef SHARK_MODELS_ONEVERSUSONE_H
#define SHARK_MODELS_ONEVERSUSONE_H


#include <shark/Models/AbstractModel.h>


namespace shark {


///
/// \brief One-versus-one Classifier.
///
/// \par
/// The one-versus-one classifier combines a number of binary
/// classifiers to form a multi-class ensemble classifier.
/// In the one-versus-one model, there exists one binary
/// classifier for each pair of classes. The predictions of
/// all binary machines are combined with a simple voting
/// scheme.
///
/// \par
/// The classifier can be extended to handle more classes on
/// the fly, without a need for re-training the existing
/// binary models.
///
template <class InputType>
class OneVersusOneClassifier : public AbstractModel<InputType, unsigned int>
{
public:
	typedef AbstractModel<InputType, unsigned int> base_type;
	typedef AbstractModel<InputType, unsigned int> binary_classifier_type;
	typedef LabeledData<InputType, unsigned int> dataset_type;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	
	/// \brief Constructor
	OneVersusOneClassifier()
	: m_classes(1)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "OneVersusOneClassifier"; }


	/// get internal parameters of the model
	virtual RealVector parameterVector() const
	{
		std::size_t total = numberOfParameters();
		RealVector ret(total);
		std::size_t used = 0;
		for (std::size_t i=0; i<m_binary.size(); i++)
		{
			std::size_t n = m_binary[i]->numberOfParameters();
			RealVectorRange(ret, Range(used, used + n)) = m_binary[i]->parameterVector();
			used += n;
		}
		return ret;
	}

	/// set internal parameters of the model
	virtual void setParameterVector(RealVector const& newParameters) {
		std::size_t used = 0;
		for (std::size_t i=0; i<m_binary.size(); i++)
		{
			std::size_t n = m_binary[i]->numberOfParameters();
			m_binary[i]->setParameterVector(ConstRealVectorRange(newParameters, Range(used, used + n)));
			used += n;
		}
		SHARK_CHECK(used == newParameters.size(),
				"[OneVersusOneClassifier::setParameterVector] invalid number of parameters");
	}

	/// return the size of the parameter vector
	virtual std::size_t numberOfParameters() const
	{
		std::size_t ret = 0;
		for (std::size_t i=0; i<m_binary.size(); i++) 
			ret += m_binary[i]->numberOfParameters();
		return ret;
	}

	/// return number of classes
	unsigned int numberOfClasses() const
	{ return m_classes; }

	/// \brief Obtain binary classifier.
	///
	/// \par
	/// The method returns the binary classifier used to distinguish
	/// class_one from class_zero. The convention class_one > class_zero
	/// is used (the inverse classifier can be constructed from this one
	/// by flipping the labels). The binary classifier outputs a value
	/// of 1 for class_one and a value of zero for class_zero.
	binary_classifier_type const& binary(unsigned int class_one, unsigned int class_zero) const
	{
		SHARK_ASSERT(class_zero < class_one);
		SHARK_ASSERT(class_one < m_classes);
		unsigned int index = class_one * (class_zero - 1) / 2 + class_zero;
		return m_binary[index];
	}

	/// \brief Add binary classifiers for one more class to the model.
	///
	/// The parameter binmodels holds a vector of n binary classifiers,
	/// where n is the current number of classes. The i-th model is this
	/// list is supposed to output a value of 1 for class n and a value
	/// of 0 for class i when faced with the binary classification problem
	/// of separating class i from class n. Afterwards the model can
	/// predict the n+1 classes {0, ..., n}.
	void addClass(std::vector<binary_classifier_type*> const& binmodels)
	{
		SHARK_CHECK(binmodels.size() == m_classes, "[OneVersusOneClassifier::addClass] wrong number of binary models");
		m_classes++;
		m_binary.insert(m_binary.end(), binmodels.begin(), binmodels.end());
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using base_type::eval;
	/// One-versus-one prediction: evaluate all binary classifiers,
	/// collect their votes, and return the class with most votes.
	void eval(
		BatchInputType const & patterns, BatchOutputType& output, State& state
	)const{
		std::size_t numPatterns = size(patterns);
		output.resize(numPatterns);
		output.clear();
		
		//matrix storing the class histogram for all patterns
		UIntMatrix votes(numPatterns,m_classes);
		votes.clear();
		
		//stores the votes of a classifier distinguishing between classes c and e
		//for all patterns
		UIntVector bin(numPatterns);
		//accumulate histograms
		for (unsigned int i=0, c=0; c<m_classes; c++)
		{
			for (std::size_t e=0; e<c; e++, i++)
			{
				m_binary[i]->eval(patterns,bin);
				for(std::size_t p = 0; p != numPatterns; ++p){
					if (bin[p] == 0) 
						votes(p,e)++; 
					else 
						votes(p,c)++;
				}
				
			}
		}
		//find the maximum class for ever pattern
		for(std::size_t p = 0; p != numPatterns; ++p){
			for (unsigned int c=1; c < m_classes; c++){
				if (votes(p,c) > votes(p,output(p))) 
					output(p) = c;
			}
		}
	}

	/// from ISerializable, reads a model from an archive
	void read(InArchive& archive)
	{
		archive & m_classes;
		archive & m_binary;
	}

	/// from ISerializable, writes a model to an archive
	void write(OutArchive& archive) const
	{
		archive & m_classes;
		//TODO: O.K. mit be leaking memory!!!
		archive & m_binary;
	}

protected:
	unsigned int m_classes;                          ///< number of classes to be distinguished
	std::vector<binary_classifier_type*> m_binary;        ///< list of binary classifiers
};


}
#endif
