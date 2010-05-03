//===========================================================================
/*!
 *  \file PopulationT.h
 *
 *  \brief Templates for typesafe uniform populations
 *
 *  \author Tobias Glasmachers
 *  \date 2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================


#ifndef _PopulationT_H_
#define _PopulationT_H_

#include <EALib/Population.h>
#include <EALib/IndividualT.h>
#include <EALib/ChromosomeT.h>


//!
//! \brief Population with uniform chromosome type CT
//!
template <typename CT>
class PopulationCT : public Population
{
public:
	PopulationCT()
	{ }

	explicit PopulationCT(unsigned n) {
		std::vector<Individual*>::resize(n);
		unsigned int i;
		for (i = n; i--;)
			*(this->begin() + i) = new IndividualCT<CT>();
	}

	//! create a population of n individuals
	//! with m chromosomes each
	PopulationCT(unsigned n, unsigned m) {
		std::vector<Individual*>::resize(n);
		unsigned int i;
		for (i=0; i<n; i++) *(this->begin() + i) = new IndividualCT<CT>(m);
	}

	PopulationCT(const IndividualCT<CT>& ind) {
		std::vector<Individual*>::resize(1);
		*begin() = new IndividualCT<CT>(ind);
	}

	PopulationCT(unsigned n, const IndividualCT<CT>& ind) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(begin() + i) = new IndividualCT<CT>(ind);
	}

	PopulationCT(unsigned n, const CT& c0) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(begin() + i) = new IndividualCT<CT>(c0);
	}

	PopulationCT(unsigned n, const CT& c0,
			   const CT& c1) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(begin() + i) = new IndividualCT<CT>(c0, c1);
	}

	PopulationCT(unsigned n, const std::vector< CT * >& v) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(begin() + i) = new IndividualCT<CT>(v);
	}

	PopulationCT(const PopulationCT<CT>& pop) {
		std::vector<Individual*>::resize(pop.size());
		for (unsigned i = pop.size(); i--;)
			*(begin() + i) = new IndividualCT<CT>(pop[ i ]);
		ascending = pop.ascending;
		spinOnce  = pop.spinOnce;
	}

	PopulationCT(const std::vector< IndividualCT<CT> * >& v)
	{
		std::vector<Individual*>::resize(v.size());
		for (unsigned i = v.size(); i--;)
			*(begin() + i) = v[i];
		subPop    = true;
	}

	~PopulationCT() { }


	IndividualCT<CT> & operator[](unsigned i) {
		return( dynamic_cast< IndividualCT<CT>& >( Population::operator[]( i ) ) );
		// return dynamic_cast<IndividualCT<CT>&>(Population::operator [] (i));
	}

	const IndividualCT<CT>& operator [ ](unsigned i) const {
		return( dynamic_cast<const IndividualCT<CT>&>(Population::operator[]( i ) ) );
	}

	PopulationCT<CT> operator()(unsigned from, unsigned to) const
	{
		RANGE_CHECK(from <= to && to < size())
		return PopulationCT<CT>(std::vector< IndividualCT<CT> * >(
							  begin() + from, begin() + to + 1));
	}

	PopulationCT<CT>& operator = (const IndividualCT<CT>& ind)
	{
		Population::operator = (ind);
		return *this;
	}

	PopulationCT<CT>& operator = (const PopulationCT<CT>& pop)
	{
		Population::operator = (pop);
		return *this;
	}

	IndividualCT<CT>& oneOfBest()
	{
		return dynamic_cast<IndividualCT<CT>&>(Population::oneOfBest());
	}

	const IndividualCT<CT>& oneOfBest() const
	{
		return dynamic_cast<const IndividualCT<CT>&>(Population::oneOfBest());
	}

	IndividualCT<CT>& best()
	{
		return dynamic_cast<IndividualCT<CT>&>(Population::best());
	}
	const IndividualCT<CT>& best() const
	{
		return dynamic_cast<const IndividualCT<CT>&>(Population::best());
	}

	IndividualCT<CT>& worst()
	{
		return dynamic_cast<IndividualCT<CT>&>(Population::worst());
	}
	const IndividualCT<CT>& worst() const
	{
		return dynamic_cast<const IndividualCT<CT>&>(Population::worst());
	}

	IndividualCT<CT>& random()
	{
		return dynamic_cast<IndividualCT<CT>&>(Population::random());
	}
	const IndividualCT<CT>& random() const
	{
		return dynamic_cast<const IndividualCT<CT>&>(Population::random());
	}

	IndividualCT<CT>& selectOneIndividual()
	{
		return dynamic_cast<IndividualCT<CT>&>(Population::selectOneIndividual());
	}
};


// workaround because
//  template <typename T> typedef PopulationCT< ChromosomeT<T> > PopulationT<T>;
// is not legal in the current C++ standard.

//!
//! \brief Population with uniform chromosome type ChromosomeT &lt; T &gt;
//!
template <typename T>
class PopulationT : public PopulationCT< ChromosomeT<T> >
{
	typedef PopulationCT< ChromosomeT<T> > super;

public:
	PopulationT()
	{ }

	explicit PopulationT(unsigned n) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(this->begin() + i) = new IndividualT<T>();
	}

	//! create a population of n individuals
	//! with m chromosomes each
	PopulationT(unsigned n, unsigned m) {
		unsigned int i;
		for (i=0; i<n; i++) *(this->begin() + i) = new IndividualT< ChromosomeT<T> >(m);
	}

	PopulationT(const IndividualT<T>& ind) {
		std::vector<Individual*>::resize(1);
		*this->begin() = new IndividualT<T>(ind);
	}

	PopulationT(unsigned n, const IndividualT<T>& ind) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(this->begin() + i) = new IndividualT<T>(ind);
	}

	PopulationT(unsigned n, const ChromosomeT<T>& c0) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(this->begin() + i) = new IndividualT<T>(c0);
	}

	PopulationT(unsigned n, const ChromosomeT<T>& c0,
			   const ChromosomeT<T>& c1) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(this->begin() + i) = new IndividualT<T>(c0, c1);
	}

	PopulationT(unsigned n, const ChromosomeT<T>& c0,
							const ChromosomeT<T>& c1, const ChromosomeT<T>& c2) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(this->begin() + i) = new IndividualT<T>(c0, c1, c2);
	}

	PopulationT(unsigned n, const std::vector< ChromosomeT<T> * >& v) {
		std::vector<Individual*>::resize(n);
		for (unsigned i = n; i--;)
			*(this->begin() + i) = new IndividualT<T>(v);
	}

	PopulationT(const PopulationT<T>& pop) {
		std::vector<Individual*>::resize(pop.size());
		for (unsigned i = pop.size(); i--;)
			*(this->begin() + i) = new IndividualT<T>(pop[ i ]);
		this->ascending = pop.ascending;
		this->spinOnce  = pop.spinOnce;
	}
	
	PopulationT(const std::vector< IndividualT<T> * >& v)
	{
		std::vector<Individual*>::resize(v.size());
		for (unsigned i = v.size(); i--;)
			*(this->begin() + i) = v[i];
		this->subPop    = true;
	}

	~PopulationT() { }


	IndividualT< T >& operator [ ](unsigned i)
	{
		return dynamic_cast<IndividualT< T >&>(super::operator [] (i));
	}

	const IndividualT< T >& operator [ ](unsigned i) const
	{
		return dynamic_cast<const IndividualT< T >&>(super::operator [] (i));
	}

	PopulationT< T > operator()(unsigned from, unsigned to) const
	{
		RANGE_CHECK(from <= to && to < this->size())
		std::vector<IndividualT<T>*> v( to-from+1 );
		for( unsigned int i = from; i <= to; i++ )
			v[i] = dynamic_cast<IndividualT<T>*>( *(this->begin() + i) );
		return( PopulationT< T >( v ) ); 
	}

	PopulationT< T >& operator = (const IndividualT< T >& ind)
	{
		super::operator = (ind);
		return *this;
	}

	PopulationT< T >& operator = (const PopulationT< T >& pop)
	{
		super::operator = (pop);
		
		return *this;
	}
	
	void append( const IndividualT<T> & ind ) {
		
		push_back( new IndividualT<T>( ind ) );
		
	}
	
	void replace(unsigned i, const IndividualT<T> & ind) {
		RANGE_CHECK(i < Population::size())
		delete *( PopulationT<T>::begin() + i );
		*( PopulationT<T>::begin() + i ) = new IndividualT<T>( ind );
	}
	
	void insert(unsigned i, const IndividualT<T> & ind) {
		RANGE_CHECK(i <= Population::size())
		std::vector< Individual * >::insert( PopulationT<T>::begin() + i, new IndividualT<T>(ind));
	}
	
	void insert(unsigned i, const PopulationT<T> & pop) {
		RANGE_CHECK(i <= Population::size())
		std::vector< Individual * >::insert( PopulationT<T>::begin() + i, pop.size(), NULL);
		for (unsigned j = pop.size(); j--;)
			*(PopulationT<T>::begin() + i + j) = new IndividualT<T>( pop[ j ] );
	}

	IndividualT< T >& oneOfBest()
	{
		return dynamic_cast<IndividualT< T >&>(super::oneOfBest());
	}

	const IndividualT< T >& oneOfBest() const
	{
		return dynamic_cast<const IndividualT< T >&>(super::oneOfBest());
	}

	IndividualT< T >& best()
	{
		return dynamic_cast<IndividualT< T >&>(super::best());
	}
	
	const IndividualT< T >& best() const
	{
		return dynamic_cast<const IndividualT< T >&>(super::best());
	}

	IndividualT< T >& worst()
	{
		return dynamic_cast<IndividualT< T >&>(super::worst());
	}
	const IndividualT< T >& worst() const
	{
		return dynamic_cast<const IndividualT< T >&>(super::worst());
	}

	IndividualT< T >& random()
	{
		return dynamic_cast<IndividualT< T >&>(super::random());
	}
	const IndividualT< T >& random() const
	{
		return dynamic_cast<const IndividualT< T >&>(super::random());
	}

	IndividualT< T >& selectOneIndividual()
	{
		return dynamic_cast<IndividualT< T >&>(super::selectOneIndividual());
	}
};


#endif
