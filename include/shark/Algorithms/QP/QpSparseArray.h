//===========================================================================
/*!
 *  \brief Special container for certain coefficients describing multi-class SVMs
 *
 *
 *  \author  T. Glasmachers
 *  \date    2011
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


#ifndef SHARK_ALGORITHMS_QP_QPSPARSEARRAY_H
#define SHARK_ALGORITHMS_QP_QPSPARSEARRAY_H


#include <shark/Algorithms/QP/QuadraticProgram.h>
#include <shark/Data/Dataset.h>


namespace shark {


/// \brief specialized container class for multi-class SVM problems
///
/// \par
/// This sparse matrix container allows to explicitly store only those
/// entries of a matrix row that deviate from a row wise default value.
/// This allows for efficient storage of the "kernel modifiers"
/// used to encode dual multi-class support vector machine problems.
template<class QpFloatType>
class QpSparseArray
{
public:
	/// \brief Non-default (non-zero) array entry.
	///
	/// \par
	/// Data structure describing a non-default
	/// (typically non-zero) entry of a row.
	struct Entry
	{
		unsigned int index;
		QpFloatType value;
	};

	/// \brief Data structure describing a row of the sparse array.
	struct Row
	{
		Entry* entry;
		unsigned int size;
		QpFloatType defaultvalue;
	};

	/// Constructor. The space parameter is an upper limit
	/// on the number of non-default (aka non-zero) entries
	/// of the array.			
	QpSparseArray(
		unsigned int height,
		unsigned int width,
		unsigned int space)
	: m_width(width)
	, m_height(height)
	, m_used(0)
	, m_data(space)
	, m_row(height)
	{
		memset(&m_row[0], 0, height * sizeof(Row));
	}

	/// number of columns
	inline unsigned int width() const
	{ return m_width; }

	/// number of rows
	inline unsigned int height() const
	{ return m_height; }

	/// obtain an element of the matrix
	QpFloatType operator () (unsigned int row, unsigned int col) const
	{
		Row const& r = m_row[row];
		unsigned int i;
		for (i=0; i<r.size; i++)
		{
			Entry const& e = r.entry[i];
			if (e.index == col) return e.value;
		}
		return r.defaultvalue;
	}

	/// obtain a row of the matrix
	inline Row const& row(unsigned int row) const
	{ return m_row[row]; }

	/// set the default value, that is, the value
	/// of all implicitly defined elements of a row
	inline void setDefaultValue(unsigned int row, QpFloatType defaultvalue)
	{ m_row[row].defaultvalue = defaultvalue; }

	/// Set a specific value. Note that entries can not
	/// be changed once they are added, and that adding
	/// elements must be done row-wise, and in order
	/// within each row. However, the order of rows does
	/// not matter.
	void add(unsigned int row, unsigned int col, QpFloatType value)
	{
		SHARK_CHECK(m_used < m_data.size(), "[QpSparseArray::add] insufficient storage space");
	
		Row& r = m_row[row];
		if (r.entry == NULL) r.entry = &m_data[m_used];
		else SHARK_CHECK(r.entry + r.size == &m_data[m_used], "[QpSparseArray::add] data must be added row-wise");
	
		m_data[m_used].index = col;
		m_data[m_used].value = value;
		m_used++;
		r.size++;
	}

protected:
	/// number of columns
	unsigned int m_width;

	/// number of rows
	unsigned int m_height;

	/// current total number of non-default components
	unsigned int m_used;

	/// storage for Entry structures
	std::vector<Entry> m_data;

	/// storage for Row structures
	std::vector<Row> m_row;
};


} // namespace shark
#endif
