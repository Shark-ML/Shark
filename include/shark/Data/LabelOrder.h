//===========================================================================
/*!
 *
 *
 * \brief       This will relabel a given dataset to have labels 0..N-1 (and vice versa)
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
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


#ifndef SHARK_LABELORDER_H
#define SHARK_LABELORDER_H

#include <shark/Core/INameable.h>
#include <shark/Core/ISerializable.h>

#include <shark/Data/Dataset.h>




namespace shark
{


/// \brief This will normalize the labels of a given dataset to 0..N-1
///
/// \par This will normalize the labels of a given dataset to 0..N-1
/// and store the ordering in a member variable.
/// After processing, the dataset will afterwards have labels ranging
/// from 0 to N-1, with N the number of classes, so usual Shark
/// trainers can work with it.
/// One can then revert the original labeling just by calling restoreOriginalLabels
class LabelOrder : public INameable
{
private:

public:


	LabelOrder() {};


	virtual ~LabelOrder() {};


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LabelOrder"; }


	/// \brief This will normalize the labels and store the ordering in the
	/// member variables. The dataset will afterwards have labels ranging
	/// from 0 to N-1, with N the number of classes.
	/// This will overwrite any previously stored label ordering in the object.
	///
	/// \param[in,out]  dataset     dataset that will be relabeled

	void normalizeLabels(LabeledData<RealVector, unsigned int> &dataset)
	{
		// determine the min and max labels of the given dataset
		int minLabel = std::numeric_limits<int>::max();
		int maxLabel = -1;
		for(std::size_t i = 0; i < dataset.numberOfElements(); ++i)
		{
			int label = dataset.labels().element(i);

			// we react allergic to negative labels
			if(label < 0)
				throw SHARKEXCEPTION("Negative label found. Will not process negative labels!");

			if(label < minLabel)
				minLabel = label;
			if(label > maxLabel)
				maxLabel = label;
		}

		// now we create an vector that can hold the label ordering
		m_labelOrder.clear();

		// and one array that tracks what we already encountered
		std::vector<int> foundLabels(maxLabel - minLabel + 1, -1);

		// and insert all labels we encounter
		size_t currentPosition = 0;
		for(std::size_t i = 0; i < dataset.numberOfElements(); i++)
		{
			// is it a new label?
			int label = dataset.labels().element(i);
			if(foundLabels[label - minLabel] == -1)
			{
				foundLabels[label - minLabel] = currentPosition;
				m_labelOrder.push_back(label);
				currentPosition++;
			}
		}

		// now map every label
		for(std::size_t i = 0; i < dataset.numberOfElements(); i++)
		{
			int label = dataset.labels().element(i);
                        dataset.labels().element(i) = foundLabels[label - minLabel];
		}
	}



	/// \brief This will restore the original labels of the dataset. This
	/// must be called with data compatible the original dataset, so that the labels will
	/// fit. The label ordering will not be destroyed after calling this function, so
	/// it can be called multiple times, e.g. to testsets or similar data.
	///
	/// \param[in,out]  dataset     dataset to relabel (restore labels)

	void restoreOriginalLabels(LabeledData<RealVector, unsigned int> &dataset)
	{
		// now map every label
		for(std::size_t i = 0; i < dataset.numberOfElements(); ++i)
		{
			int label = dataset.labels().element(i);

			// check if the reordering fit the data
			if(label >= (int) m_labelOrder.size())
				throw SHARKEXCEPTION("Dataset labels does not fit to the stored ordering!");

			// relabel
			label = m_labelOrder[label];
			dataset.labels().element(i) = label;
		}
	}



	/// \brief Get label ordering directly
	///
	/// \param[out] labelOrder      vector to store the current label order.

	void getLabelOrder(std::vector<int> &labelOrder)
	{
		labelOrder = m_labelOrder;
	}



	/// \brief Get label ordering directly
	///
	/// \param[out] labelOrder      vector to store the current label order.

        void getLabelOrder (std::vector<unsigned int> &labelOrder)
        {
                labelOrder = std::vector<unsigned int>( m_labelOrder.begin(), m_labelOrder.end() );
        }


	/// \brief Set label ordering directly
	///
	/// \param[in] labelOrder      vector with the new label order

	void setLabelOrder(std::vector<int> &labelOrder)
	{
		m_labelOrder = labelOrder;
	}


        /// \brief Set label ordering directly
        ///
        /// \param[in] labelOrder      vector with the new label order
        void setLabelOrder (std::vector<unsigned int> &labelOrder)
        {
                m_labelOrder  = std::vector<int>( labelOrder.begin(), labelOrder.end() );
        }


protected:

	std::vector<int> m_labelOrder;
};

}

#endif

