//===========================================================================
/*!
 *  \file ChromosomeT.h
 *
 *  \author  Martin Kreutz
 *  \date    1995-01-01
 *
 *  \brief base class for chromosomes that are of the template type std::vector
 *
 *  \par Copyright (c) 1995-2003:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      EALib
 *  <BR>
 * 
 *
 *  <BR><HR>
 *  This file is part of EALib. This library is free software;
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


#ifndef __CHROMOSOMET_H
#define __CHROMOSOMET_H

#ifdef _WIN32
// disable warning C4786: symbol length > 255 character,
// okay to ignore
#pragma warning(disable: 4786)
#endif

#include <algorithm>
#include <typeinfo>
#include <cstring>

#include <SharkDefs.h>
#include <Rng/GlobalRng.h>
#include <EALib/Interval.h>
#include <EALib/Chromosome.h>
#include <EALib/PVMinterface.h>


//===========================================================================
//!
//! \brief base class for chromosomes that are of the template type std::vector
//!
template < class T >
class ChromosomeT_base : public Chromosome, public std::vector< T >
{
public:
	ChromosomeT_base()
	{ }
	explicit ChromosomeT_base(unsigned    l) : std::vector< T >(l)
	{ }
	ChromosomeT_base(unsigned l, const T& v) : std::vector< T >(l, v)
	{ }
	ChromosomeT_base(const std::vector< T >&   v) : std::vector< T >(v)
	{ }

	const char* typeOfAlleles() const
	{
		return typeid(T).name();
	}
	unsigned    sizeOfAlleles() const
	{
		return sizeof(T);
	}

	unsigned    size() const
	{
		return static_cast< const std::vector< T > * >(this)->size();
	}

	Chromosome& operator = (const Chromosome& c)
	{
		static_cast< std::vector< T > * >(this)->operator = (dynamic_cast< const std::vector< T >& >(c));
		return *this;
	}

	Chromosome& operator = (const std::vector< T >& c)
	{
		static_cast< std::vector< T > * >(this)->operator = (c);
		return *this;
	}

	//
	// disambiguate the other two assignment operators
	//
	Chromosome& operator = (const ChromosomeT_base< T >& c)
	{
		static_cast< std::vector< T > * >(this)->operator = (c);
		return *this;
	}

	Chromosome& operator = (const T& c)
	{
		for (unsigned i = this->size(); i--; (*this)[ i ] = c);
		return *this;
	}

	//=======================================================================
	//
	// change size of a chromosome
	//
	void resize(unsigned n)
	{
		if (n < size())
		{
			static_cast< std::vector< T > * >(this)->erase(this->begin() + n, this->end());
		}
		else if (n > size())
		{
			static_cast< std::vector< T > * >(this)->insert(this->end(), n - size(), T());
		}
	}

	//=======================================================================
	//
	// duplicates circulary a sequence of alleles
	// chunks must not overlap
	//
	void duplicate(unsigned start,
				   unsigned stop,
				   unsigned dest)
	{
		RANGE_CHECK(start < size() &&
					stop  < size() &&
					dest  < size())

		if (start > stop) stop += size();

		for (unsigned i = start, j = dest; i <= stop; i++, j++)
			(*this)[ j % size()] = (*this)[ i % size()];
	}


	//=======================================================================
	//
	//
	//
	void invert(unsigned start,
				unsigned stop,
				unsigned granularity = 1)
	{
		unsigned  i, j, k, l;

		RANGE_CHECK(start < size() &&
					stop  < size())

		if (start > stop) stop += size();

		i = start;
		j = stop - granularity + 1;
		l = (stop - start + 1) / (2 * granularity);

		while (l--)
		{
			for (k = granularity; k--;)
				swap((i + k) % size(), (j + k) % size());
			i += granularity;
			j -= granularity;
		}
	}


	//=======================================================================
	//
	//
	//
	void invert(unsigned granularity = 1)
	{
		invert(0, size() - 1, granularity);
	}


	//=======================================================================
	//
	//
	//
	void transcribe(unsigned start,
					unsigned stop,
					const Chromosome& chrom)
	{
		const std::vector< T >& c = dynamic_cast< const std::vector< T >& >(chrom);

		RANGE_CHECK(start < c.size() &&
					stop  < c.size())

		if (start > stop) stop += c.size();

		resize(stop - start + 1);

		for (unsigned i = start, j = 0; i <= stop; i++, j++)
			(*this)[ j ] = c[ i % c.size()];
	}


	//=======================================================================
	//
	//
	//
	void swap(unsigned i, unsigned j)
	{
		RANGE_CHECK(i < size() && j < size())
#ifdef __NO_BITPACKING__
		std::swap((*this)[ i ], (*this)[ j ]);
#else
		//
		// STL swap doesn't work for vector< bool >
		// on some systems
		//
		T t            = (*this)[ i ];
		(*this)[ i ] = (*this)[ j ];
		(*this)[ j ] = t;
#endif
	}


	//=======================================================================
	//
	//
	//
	void shuffle()
	{
		for (unsigned i = this->size(); i--;)
			swap(i, Rng::discrete(0, size() - 1));
	}


	//=======================================================================
	//
	//
	//
	void replace(unsigned i, const T& v)
	{
		RANGE_CHECK(i < size())
		(*this)[ i ] = v;
	}


	//=======================================================================
	//
	//
	//
	void replace(unsigned i, const Chromosome& chrom)
	{
		const std::vector< T >& v = dynamic_cast< const std::vector< T >& >(chrom);
		RANGE_CHECK(i + v.size() <= size())
		std::copy(v.begin(), v.end(), this->begin() + i);
	}


	//=======================================================================
	//
	//
	//
	void insert(unsigned i, const T& allele)
	{
		RANGE_CHECK(i <= size())
		std::vector< T >::insert(this->begin() + i, 1, allele);
	}


	//=======================================================================
	//
	//
	//
	void insert(unsigned i, const Chromosome& chrom)
	{
		const std::vector< T >& v = dynamic_cast< const std::vector< T >& >(chrom);
		RANGE_CHECK(i <= size())
		static_cast< std::vector< T > * >(this)->insert(this->begin() + i, v.begin(), v.end());
	}


	//=======================================================================
	//
	//
	//
	void append(const T& allele)
	{
		std::vector< T >::push_back(allele);
	}


	//=======================================================================
	//
	//
	//
	void append(const Chromosome& chrom)
	{
		const std::vector< T >& v = dynamic_cast< const std::vector< T >& >(chrom);
		static_cast< std::vector< T > * >(this)->insert(this->end(), v.begin(), v.end());
	}


	//=======================================================================
	//
	//
	//
	void remove(unsigned i)
	{
		RANGE_CHECK(i < size())
		static_cast< std::vector< T > * >(this)->erase(this->begin() + i);
	}


	//=======================================================================
	//
	//
	//
	void remove(unsigned i, unsigned k)
	{
		if (i <= k)
		{
			RANGE_CHECK(k < size())
			static_cast< std::vector< T > * >(this)->erase(this->begin() + i, this->begin() + k);
		}
	}


	//=======================================================================
	//
	// rotation operators
	//
	//=======================================================================
	//
	//
	//
	void rotateRight(unsigned n = 1)
	{
		//
		// can't rotate vector< bool > if they are represented by a packed
		// bit string
		//
		n = n % size();
#ifdef __NO_BITPACKING__
		std::rotate(this->begin(), this->end() - n, this->end());
#else
		std::cerr << "cannot rotate bool chromosome represented as bit string by "
		<< n << " bits!" << std::endl;
#endif
	}


	//=======================================================================
	//
	//
	//
	void rotateLeft(unsigned n = 1)
	{
		//
		// can't rotate vector< bool > if they are represented by a packed
		// bit string
		//
		n = n % size();
#ifdef __NO_BITPACKING__
		std::rotate(begin(), begin() + n, end());
#else
		std::cerr << "cannot rotate bool chromosome represented as bit string by "
		<< n << " bits!" << std::endl;
#endif
	}


	//=======================================================================
	//
	//
	//
#ifndef _WIN32
	//
	// Visual C++ 5.0 can't distinguish between different instantiations of
	// template class vector< T >
	void crossover(const Chromosome& dadChrom,
				   const Chromosome& momChrom,
				   const std::vector< unsigned >& points)
	{
		SIZE_CHECK(dadChrom.size() == momChrom.size())

		resize(dadChrom.size());

		if (size() > 0)
		{
			unsigned i, j, p, max, swp;
			const std::vector< T >& dad = dynamic_cast< const std::vector< T >& >(dadChrom);
			const std::vector< T >& mom = dynamic_cast< const std::vector< T >& >(momChrom);

			//
			// sort crossover points
			//
			std::vector< unsigned > pts(points);
			for (i = pts.size(); i--;)
			{
				for (max = pts[ i ], p = j = i; j--;)
					if (pts[ j ] > max) p = j;
				RANGE_CHECK(max <= size())
				if (p != i) std::swap(pts[ i ], pts[ p ]);
			}

			for (swp = i = j = 0; i < pts.size(); i++, swp ^= 1)
			{
				if (swp)
					for (; j < pts[ i ]; j++)
						(*this)[ j ] = mom[ j ];
				else
					for (; j < pts[ i ]; j++)
						(*this)[ j ] = dad[ j ];
			}

			if (swp)
				for (; j < size(); j++)
					(*this)[ j ] = mom[ j ];
			else
				for (; j < size(); j++)
					(*this)[ j ] = dad[ j ];
		}
	}


	//=======================================================================
	//
	//
	//
	void crossover(Chromosome& mate, const std::vector< unsigned >& points)
	{
		SIZE_CHECK(this->size() == mate.size())

		if (size() > 0)
		{
			unsigned    i, j, p, max, swp;
			std::vector< T >& mate1 = *this;
			std::vector< T >& mate2 = dynamic_cast< std::vector<T>& >(mate);

			//
			// sort crossover points
			//
			std::vector< unsigned > pts(points);
			for (i = pts.size(); i--;)
			{
				for (max = pts[ i ], p = j = i; j--;)
					if (pts[ j ] > max) p = j;
				RANGE_CHECK(max <= size())
				if (p != i) std::swap(pts[ i ], pts[ p ]);
			}

			for (swp = pts.size() & 1, i = j = 0; i < pts.size(); i++, swp ^= 1)
			{
				if (swp)
					for (; j < pts[ i ]; j++)
					{
#ifdef __NO_BITPACKING__
						std::swap(mate1[ j ], mate2[ j ]);
#else
						//
						// STL swap doesn't work for vector< bool >
						// on some systems
						//
						T t        = mate1[ j ];
						mate1[ j ] = mate2[ j ];
						mate2[ j ] = t;
#endif
					}
				else
					j = pts[ i ];
			}
		}
	}
#endif // !_WIN32


	//=======================================================================
	//
	//
	//
	void crossover(const Chromosome& dadChrom,
				   const Chromosome& momChrom,
				   const std::vector< bool >& pos)
	{
		SIZE_CHECK(dadChrom.size() == momChrom.size())

		resize(dadChrom.size());

		if (size())
		{
			const std::vector< T >& dad = dynamic_cast< const std::vector< T >& >(dadChrom);
			const std::vector< T >& mom = dynamic_cast< const std::vector< T >& >(momChrom);

			unsigned i, m;
			bool swp = false;
			for (i = 0, m = Shark::min(size(), (unsigned) pos.size()); i < m; i++)
			{
				// crossover point => swap parents

				// Visual C++ 5.0 doesn't like xor between bools
				//swp ^= pos[ i ];
				swp = swp != pos[ i ];
				(*this)[ i ] = swp ? mom[ i ] : dad[ i ];
			}
		}
	}


	//=======================================================================
	//
	//
	//
	void crossover(Chromosome& mate, const std::vector< bool >& pos)
	{
		SIZE_CHECK(this->size() == mate.size())

		if (size())
		{
			std::vector< T >& mate1 = *this;
			std::vector< T >& mate2 = dynamic_cast< std::vector<T>& >(mate);

			unsigned i, m;
			bool swp = false;
			for (i = 0, m = Shark::min(size(), (unsigned) pos.size()); i < m; ++i)
				//if( swp ^= pos[ i ] )
				if( ( swp = ( swp != pos[ i ] ) ) )
				{
#ifdef __NO_BITPACKING__
					std::swap(mate1[ i ], mate2[ i ]);
#else
					//
					// STL swap doesn't work for vector< bool >
					// on some systems
					//
					T t        = mate1[ i ];
					mate1[ i ] = mate2[ i ];
					mate2[ i ] = t;
#endif
				}
		}
	}


	//=======================================================================
	//
	//
	//
	void crossover(const Chromosome& dadChrom,
				   const Chromosome& momChrom,
				   unsigned npoints,
				   unsigned align = 1,
				   bool chromswap = 0)
	{
		std::vector< bool > pos(dadChrom.size(), false);
		unsigned startpos = chromswap ? 0 : 1;

#ifdef __NO_BITPACKING__
		for (unsigned i = 0; i < npoints; i++)
			pos[ Rng::discrete(startpos,(pos.size()-1)/align)*align ] ^= true;
#else
		//
		// vector< bool > is now represented by a packed bit array
		//
		for (unsigned i = 0; i < npoints; i++)
		{
			unsigned j = Rng::discrete(startpos, (pos.size() - 1) / align) * align;
			pos[ j ] = ! pos[ j ];
		}
#endif

		crossover(dadChrom, momChrom, pos);
	}


	//=======================================================================
	//
	//
	//
	void crossover(Chromosome& mate,
				   unsigned npoints,
				   unsigned align = 1,
				   bool chromswap = 0)
	{
		std::vector< bool > pos(mate.size(), false);
		unsigned startpos = chromswap ? 0 : 1;

#ifdef __NO_BITPACKING__
		for (unsigned i = 0; i < npoints; i++)
			pos[ Rng::discrete(startpos,(pos.size()-1)/align)*align] ^= true;
#else
		//
		// vector< bool > is now represented by a packed bit array
		//
		for (unsigned i = 0; i < npoints; i++)
		{
			unsigned j = Rng::discrete(startpos, (pos.size() - 1) / align) * align;
			pos[ j ] = ! pos[ j ];
		}
#endif

		crossover(mate, pos);
	}


	//=======================================================================
	//
	//
	//
	void crossover(const Chromosome& dadChrom,
				   const Chromosome& momChrom,
				   const Chromosome& posChrom)
	{
		crossover(dadChrom, momChrom,
				  dynamic_cast< const std::vector< bool >& >(posChrom));
	}


	//=======================================================================
	//
	//
	//
	void crossover(Chromosome& mate, const Chromosome& posChrom)
	{
		crossover(mate,
				  dynamic_cast< const std::vector< bool >& >(posChrom));
	}


	//=======================================================================
	//
	//
	//
	void crossoverUniform(const Chromosome& dadChrom,
						  const Chromosome& momChrom,
						  const std::vector< bool >& pos)
	{
		SIZE_CHECK(dadChrom.size() == momChrom.size())

		resize(dadChrom.size());

		if (size() > 0)
		{
			const std::vector< T >& dad = dynamic_cast< const std::vector< T >& >(dadChrom);
			const std::vector< T >& mom = dynamic_cast< const std::vector< T >& >(momChrom);

			for (unsigned i = Shark::min(size(), (unsigned) pos.size()); i--;)
				(*this)[ i ] = pos[ i ] ? mom[ i ] : dad[ i ];
		}
	}


	//=======================================================================
	//
	//
	//
	void crossoverUniform(Chromosome& mateChrom,
						  const std::vector< bool >& pos)
	{
		SIZE_CHECK(this->size() == mateChrom.size())

		if (size())
		{
			std::vector< T >& mate = dynamic_cast< std::vector< T >& >(mateChrom);

			for (unsigned i = Shark::min(size(), (unsigned) pos.size()); i--;)
				if (pos[ i ])
				{
#ifdef __NO_BITPACKING__
					std::swap((*this)[ i ], mate[ i ]);
#else
					//
					// STL swap doesn't work for vector< bool >
					// on some systems
					//
					T t = (*this)[ i ];
					(*this)[ i ] = mate[ i ];
					mate[ i ] = t;
#endif
				}
		}
	}


	//=======================================================================
	//
	//
	//
	void crossoverUniform(const Chromosome& dadChrom,
						  const Chromosome& momChrom)
	{
		SIZE_CHECK(dadChrom.size() == momChrom.size())

		resize(dadChrom.size());

		if (size() > 0)
		{
			const std::vector< T >& dad = dynamic_cast< const std::vector< T >& >(dadChrom);
			const std::vector< T >& mom = dynamic_cast< const std::vector< T >& >(momChrom);

			for (unsigned i = this->size(); i--;)
				(*this)[ i ] = Rng::coinToss(0.5) ? dad[ i ] : mom[ i ];
		}
	}


	//=======================================================================
	//
	//
	//
	void crossoverUniform(Chromosome& mateChrom)
	{
		SIZE_CHECK(this->size() == mateChrom.size())

		if (size())
		{
			std::vector< T >& mate = dynamic_cast< std::vector< T >& >(mateChrom);

			for (unsigned i = size(); i--;)
				if (Rng::coinToss(0.5))
				{
#ifdef __NO_BITPACKING__
					std::swap((*this)[ i ], mate[ i ]);
#else
					//
					// STL swap doesn't work for vector< bool >
					// on some systems
					//
					T t = (*this)[ i ];
					(*this)[ i ] = mate[ i ];
					mate[ i ] = t;
#endif
				}
		}
	}


	//=======================================================================
	//
	//
	//
	void crossoverUniform(const Chromosome& dadChrom,
						  const Chromosome& momChrom,
						  const Chromosome& posChrom)
	{
		crossoverUniform(dadChrom, momChrom,
						 dynamic_cast< const std::vector< bool >& >(posChrom));
	}


	//=======================================================================
	//
	//
	//
	void crossoverUniform(Chromosome& mateChrom,
						  const Chromosome& posChrom)
	{
		crossoverUniform(mateChrom,
						 dynamic_cast< const std::vector< bool >& >(posChrom));
	}


	//=======================================================================
	//
	//
	//
	void recombineDiscrete(const Chromosome& dad,
						   const Chromosome& mom)
	{
		crossoverUniform(dad, mom);
	}


	//=======================================================================
	//
	//
	//
	void recombineDiscrete(Chromosome& mate)
	{
		crossoverUniform(mate);
	}

#ifndef __NO_GENERIC_IOSTREAM
	//=======================================================================
	//
	//
	//
	void writeTo(std::ostream& os) const
	{
		os	<< "ChromosomeT<"
		<< typeid(T).name()
		<< ">("
		<< this->size() << ")"
		<< std::endl;
		for (unsigned i = 0; i < size(); i++)
		{
			if (i)
			{
				os << '\t';
			}
			os << (*this)[ i ];
		}
		os << std::endl;
	}

	//=======================================================================
	//
	//
	//
	void readFrom(std::istream& is)
	{
		std::string s;
//		is.getline( s );
		is >> s;
		is.get();    // skip end of line

		if (is.good() &&
				s.substr(0, 12) == "ChromosomeT<" &&
				s.find('>') != std::string::npos &&
				s.substr(12, s.find('>') - 12) == typeid(T).name())
		{

			resize(atoi(s.substr(s.find('>') + 2).c_str()));
			for (unsigned i = 0; i < size(); i++)
			{
				T t;
				is >> t; // bit-reference vector< bool > !!!
				(*this)[ i ] = t;
			}
		}
		else
#if (defined(__GNUC__) && __GNUC__ > 2)
			//is.setf( std::ios::fmtflags(std::__ios_flags::_S_failbit) );
			is.setf(std::ios::fmtflags(std::ios::failbit));
#else
			is.setf(std::ios::failbit);
#endif
	}
#endif // !__NO_GENERIC_IOSTREAM

protected:
	Chromosome* clone() const
	{
		return new ChromosomeT_base< T >(*this);
	}
	Chromosome* empty() const
	{
		return new ChromosomeT_base< T >;
	}
};

//===========================================================================
//!
//! \brief template class for chromosomes
//!
template < class T >
class ChromosomeT : public ChromosomeT_base< T >
{
public:
	ChromosomeT()
	{ }
	explicit ChromosomeT(unsigned    l)
			: ChromosomeT_base< T >(l)
	{ }
	ChromosomeT(unsigned l, const T& v)
			: ChromosomeT_base< T >(l, v)
	{ }
	ChromosomeT(const std::vector< T >&   v)
			: ChromosomeT_base< T >(v)
	{ }

protected:
	Chromosome* clone() const
	{
		return new ChromosomeT< T >(*this);
	}
	Chromosome* empty() const
	{
		return new ChromosomeT< T >;
	}

#ifndef __NO_GENERIC_IOSTREAM
	//=======================================================================
	//
	//
	//
	void writeTo(std::ostream& os) const
	{
		os << "ChromosomeT<" << typeid(T).name() << ">(" << this->size() << ")" << std::endl;
		for (unsigned i = 0; i < this->size(); i++)
		{
			if (i) os << '\t';
			os << (*this)[ i ];
		}
		os << std::endl;
	}

	//=======================================================================
	//
	//
	//
	void readFrom(std::istream& is)
	{
		std::string s;
		is >> s;
		is.get();    // skip end of line

		if (is.good() &&
				s.substr(0, 12) == "ChromosomeT<" &&
				s.find('>') != std::string::npos &&
				s.substr(12, s.find('>') - 12) == typeid(T).name())
		{

			((std::vector<T>*)this)->resize(atoi(s.substr(s.find('>') + 2).c_str()));
			for (unsigned i = 0; i < this->size(); i++)
				is >> (*this)[ i ];
		}
		else
#if (defined(__GNUC__) && __GNUC__ > 2)
			// is.setf( std::ios::fmtflags(std::__ios_flags::_S_failbit) );
			is.setf(std::ios::fmtflags(std::ios::failbit));
#else
			is.setf(std::ios::failbit);
#endif
	}
#endif // !__NO_GENERIC_IOSTREAM

};

//===========================================================================
//!
//! \brief template class for chromosomes based on numeric types
//!
template < class T >
class ChromosomeT_num : public ChromosomeT_base< T >
{
private:
	static void initialize(T& v, T min, T max)
	{
		if (T(0.5) == 0.5)    // continuous
			v = T(Rng::uni(double(min), double(max)));
		else // discrete
			v = T(Rng::discrete(long(min), long(max)));
	}

	static void cutOff(T& v, T min, T max)
	{
		if (v < min) v = min;
		if (v > max) v = max;
	}

public:
	ChromosomeT_num()
	{ }
	explicit ChromosomeT_num(unsigned l)
			: ChromosomeT_base< T >(l)
	{ }
	ChromosomeT_num(unsigned l, const T& v)
			: ChromosomeT_base< T >(l, v)
	{ }
	ChromosomeT_num(const std::vector< T >& v)
			: ChromosomeT_base< T >(v)
	{ }


	//=======================================================================
	//
	//
	//
	void initialize(T min, T max)
	{
		for (unsigned i = this->size(); i--;)
			initialize((*this)[ i ], min, max);
	}


	//=======================================================================
	//
	//
	//
// 	void initialize(const Chromosome& minChrom,
// 					const Chromosome& maxChrom)
// 	{
// 		SIZE_CHECK(minChrom.size() == maxChrom.size())
// 
// 		((std::vector<T>*)this)->resize(minChrom.size());
// 
// 		if (this->size())
// 		{
// 			const std::vector< T >& min = dynamic_cast< const std::vector< T >& >(minChrom);
// 			const std::vector< T >& max = dynamic_cast< const std::vector< T >& >(maxChrom);
// 
// 			for (unsigned i = this->size(); i--;)
// 				initialize((*this)[ i ], min[ i ], max[ i ]);
// 		}
// 	}


	void initialize(const std::vector<T>& min,
					const std::vector<T>& max)
	{
		SIZE_CHECK(min.size() == max.size())

		((std::vector<T>*)this)->resize(min.size());

		for (unsigned i = this->size(); i--;)
			initialize((*this)[i], min[i], max[i]);
	}


	//=======================================================================
	//
	//
	//
	void cutOff(T min, T max)
	{
		for (unsigned i = this->size(); i--;)
			cutOff((*this)[ i ], min, max);
	}


	//=======================================================================
	//
	//
	//
	void cutOff(const Chromosome& minChrom,
				const Chromosome& maxChrom)
	{
		SIZE_CHECK(minChrom.size() == maxChrom.size())

		SIZE_CHECK((*this).size() == minChrom.size())

		if (this->size())
		{
			const std::vector< T >& min = dynamic_cast< const std::vector< T >& >(minChrom);
			const std::vector< T >& max = dynamic_cast< const std::vector< T >& >(maxChrom);

			for (unsigned i = this->size(); i--;)
				cutOff((*this)[ i ], min[ i ], max[ i ]);
		}
	}


	//=======================================================================
	//
	//
	//
	void mutateUniform(T min, T max, double p)
	{
		for (unsigned i = this->size(); i--;)
			if (Rng::coinToss(p))
				initialize((*this)[ i ], min, max);
	}


	//=======================================================================
	//
	//
	//
	void mutateUniform(T min, T max,
					   const std::vector< double >& p,
					   bool cycle = false)
	{
		RANGE_CHECK(p.size() <= this->size())

		for (unsigned i = cycle ? this->size() : p.size(); i--;)
			if (Rng::coinToss(p[ i % p.size()]))
				initialize((*this)[ i ], min, max);
	}


	//=======================================================================
	//
	//
	//
	void mutateUniform(T min, T max,
					   const Chromosome& p,
					   bool cycle = false)
	{
		mutateUniform(min, max,
					  dynamic_cast< const std::vector< double >& >(p),
					  cycle);
	}


	//=======================================================================
	//
	//
	//
	void mutateUniform(const Chromosome& minChrom,
					   const Chromosome& maxChrom,
					   double p)
	{
		SIZE_CHECK(this->size() == minChrom.size())
		SIZE_CHECK(this->size() == maxChrom.size())

		if (this->size() > 0)
		{
			const std::vector< T >& min = dynamic_cast< const std::vector< T >& >(minChrom);
			const std::vector< T >& max = dynamic_cast< const std::vector< T >& >(maxChrom);

			for (unsigned i = this->size(); i--;)
				if (Rng::coinToss(p))
					initialize((*this)[ i ], min[ i ], max[ i ]);
		}
	}


	//=======================================================================
	//
	//
	//
	void mutateUniform(const Chromosome& minChrom,
					   const Chromosome& maxChrom,
					   const std::vector< double >& p,
					   bool cycle = false)
	{
		SIZE_CHECK(this->size() == minChrom.size())
		SIZE_CHECK(this->size() == maxChrom.size())
		RANGE_CHECK(p.size() <= this->size())

		if (this->size())
		{
			const std::vector< T >& min = dynamic_cast< const std::vector< T >& >(minChrom);
			const std::vector< T >& max = dynamic_cast< const std::vector< T >& >(maxChrom);

			for (unsigned i = cycle ? this->size() : p.size(); i--;)
				if (Rng::coinToss(p[ i % p.size()]))
					initialize((*this)[ i ], min[ i ], max[ i ]);
		}
	}


	//=======================================================================
	//
	//
	//
	void mutateUniform(const Chromosome& min,
					   const Chromosome& max,
					   const Chromosome& p,
					   bool cycle = false)
	{
		mutateUniform(min, max,
					  dynamic_cast< const std::vector< double >& >(p),
					  cycle);
	}


	bool operator == (const Chromosome& c) const
	{
		//
		// this annoying static cast is necessary to satisfy
		// the pedantic VC++ 5.0
		//
		return static_cast< const std::vector< T >& >(*this) == dynamic_cast< const std::vector< T >& >(c);
	}

	bool operator < (const Chromosome& c) const
	{
		return static_cast< const std::vector< T >& >(*this) <  dynamic_cast< const std::vector< T >& >(c);
	}
};

//===========================================================================
//!
//! \brief chromosome based on double numbers
//!
template < >
class ChromosomeT< double > : public ChromosomeT_num< double >
{
public:
	ChromosomeT()
	{ }
	explicit ChromosomeT(unsigned l)
			: ChromosomeT_num< double >(l)
	{ }
	ChromosomeT(unsigned l, const double& v)
			: ChromosomeT_num< double >(l, v)
	{ }
	ChromosomeT(const std::vector< double >& v)
			: ChromosomeT_num< double >(v)
	{ }
	~ChromosomeT();

	void        initializeRotate(const ChromosomeT< double >&);
	void        initializeRotate(const std::vector< double >&);
	void        initializeRotate(double, double);

	void        showRotate();

	void        decodeBinary(const Chromosome& chrom,
							 const Interval&   range,
							 unsigned          nbits,
							 bool              useGray = false);

	void        accumulate(const std::vector< double >&, double);
	void        accumulate(const Chromosome&, double);

	void        mutateCauchy(double scale);
	void        mutateCauchy(const std::vector< double >& scale, bool = false);
	void        mutateCauchy(const Chromosome& scale, bool = false);
	void        mutateCauchy(const ChromosomeT< double >& scale, bool = false);

	void        mutateNormal(double variance);
	void        mutateNormal(const std::vector< double >& variances, bool = false);
	void        mutateNormal(const Chromosome& variances, bool = false);
	void        mutateNormal(const ChromosomeT< double >& variances, bool = false);

	void        mutateNormalRotAngles(const std::vector< double >&, const std::vector< double >&);
	void        mutateNormalRotAngles(const Chromosome&, const Chromosome&);

	void        mutateLogNormal(double, double);
	void        mutateMSR(double);

	void        mutateRotate(ChromosomeT<double> &);
	void        mutateRotate(ChromosomeT<double> &, double, double, double, int, double);

	void        recombineIntermediate(const Chromosome&, const Chromosome&);
	void        recombineGenIntermediate(const Chromosome&, const Chromosome&);
	void        recombineGeomIntermediate(const Chromosome&, const Chromosome&);

	void        recombineIntermediate(Chromosome&);
	void        recombineGenIntermediate(Chromosome&);
	void        recombineGeomIntermediate(Chromosome&);


	void        SBX(ChromosomeT< double >&, double, double = .5);
	void        SBX(ChromosomeT< double >&, double, double, double, double = .5, double  = 1.E-12);
	void        SBX(ChromosomeT< double >&, std::vector<double > &, std::vector<double > &, double, double = .5, double  = 1.E-12);

	void        simpleMutatePolynomial(double, double, double, double);
	void        simpleMutatePolynomial(std::vector<double > &, std::vector<double > &, double, double);
	void        mutatePolynomial(double, double, double, double);
	void        mutatePolynomial(std::vector<double > &, std::vector<double > &, double, double);

protected:
	Chromosome* clone() const
	{
		return new ChromosomeT< double >(*this);
	}
	Chromosome* empty() const
	{
		return new ChromosomeT< double >;
	}

	/*! Part of PVM-send routine for type double Chromosomes */

	int pvm_pkchrom()
	{
		//cout << "\t  pk_chrom_double" << endl;
		unsigned i;

		unsigned *s = new unsigned;
		*s = this->size();
		pvm_pkuint(s, 1, 1);
		delete s;

		double *u = new double[this->size()];
		for (i = 0; i < this->size(); i++)
			u[i] = (*this)[i];
		pvm_pkdouble(u, this->size(), 1);
		delete[] u;

		return 1;
	};

	/*! Part of PVM-receive routine for type double Chromosomes */

	int pvm_upkchrom()
	{
		//cout << "\t  upk_chrom_double" << endl;
		unsigned i;

		unsigned *s = new unsigned;
		pvm_upkuint(s, 1, 1);
		(*this).resize(*s);
		delete s;

		double *u = new double[this->size()];
		pvm_upkdouble(u, this->size(), 1);
		for (i = 0; i < this->size(); i++)
			(*this)[i] = u[i];
		delete[] u;

		return 1;
	};


#ifndef __NO_GENERIC_IOSTREAM
	void writeTo(std::ostream& os) const
	{
		os << "ChromosomeT<" << typeid(double).name() << ">(" << size() << ")" << std::endl;
		for (unsigned i = 0; i < size(); i++)
		{
			if (i) os << '\t';
			os << (*this)[ i ];
		}
		os << std::endl;
	}

	void readFrom(std::istream& is)
	{
		std::string s;
		//is.getline( s );
		is >> s;
		is.get();    // skip end of line

		if (is.good() &&
				s.substr(0, 12) == "ChromosomeT<" &&
				s.find('>') != std::string::npos &&
				s.substr(12, s.find('>') - 12) == typeid(double).name())
		{

			resize(atoi(s.substr(s.find('>') + 2).c_str()));
			for (unsigned i = 0; i < size(); i++)
				is >> (*this)[ i ];
		}
		else
#if (defined(__GNUC__) && __GNUC__ > 2)
			// is.setf( std::ios::fmtflags(std::__ios_flags::_S_failbit) );
			is.setf(std::ios::fmtflags(std::ios::failbit));
#else
			is.setf(std::ios::failbit);
#endif
	}

#endif // !__NO_GENERIC_IOSTREAM

};

//===========================================================================
//!
//! \brief chromosome based on characters
//!
template < >
class ChromosomeT< char > : public ChromosomeT_num< char >
{
public:
	ChromosomeT()
	{ }
	explicit ChromosomeT(unsigned l)
			: ChromosomeT_num< char >(l)
	{ }
	ChromosomeT(unsigned l, const char& v)
			: ChromosomeT_num< char >(l, v)
	{ }
	ChromosomeT(const std::vector< char >& v)
			: ChromosomeT_num< char >(v)
	{ }

protected:
	Chromosome* clone() const
	{
		return new ChromosomeT< char >(*this);
	}
	Chromosome* empty() const
	{
		return new ChromosomeT< char >;
	}

	/*! Part of PVM-send routine for type char Chromosomes */

	int pvm_pkchrom()
	{
		//cout << "\t  pk_chrom_char" << endl;
		unsigned i;

		unsigned *s = new unsigned;
		*s = this->size();
		pvm_pkuint(s, 1, 1);
		delete s;

		char *u = new char[this->size()];
		for (i = 0; i < this->size(); i++)
			u[i] = (*this)[i];
		pvm_pkbyte(u, this->size(), 1);
		delete[] u;

		return 1;
	};


	/*! Part of PVM-receive routine for type char Chromosomes */

	int pvm_upkchrom()
	{
		//cout << "\t  upk_chrom_char" << endl;
		unsigned i;

		unsigned *s = new unsigned;
		pvm_upkuint(s, 1, 1);
		(*this).resize(*s);
		delete s;

		char *u = new char[this->size()];
		pvm_upkbyte(u, this->size(), 1);
		for (i = 0; i < this->size(); i++)
			(*this)[i] = u[i];
		delete[] u;

		return 1;
	};

#ifndef __NO_GENERIC_IOSTREAM
	void writeTo(std::ostream& os) const
	{
		os << "ChromosomeT<" << typeid(char).name() << ">(" << size() << ")" << std::endl;
		for (unsigned i = 0; i < size(); i++)
			os.put((*this)[ i ]);
		os << std::endl;
	}

	void readFrom(std::istream& is)
	{
		std::string s;
		//is.getline( s );
		is >> s;
		is.get();    // skip end of line

		if (is.good() &&
				s.substr(0, 12) == "ChromosomeT<" &&
				s.find('>') != std::string::npos &&
				s.substr(12, s.find('>') - 12) == typeid(char).name())
		{

			resize(atoi(s.substr(s.find('>') + 2).c_str()));
			for (unsigned i = 0; i < size(); i++)
				is.get((*this)[ i ]);
		}
		else
#if (defined(__GNUC__) && __GNUC__ > 2)
			// is.setf( std::ios::fmtflags(std::__ios_flags::_S_failbit) );
			is.setf(std::ios::fmtflags(std::ios::failbit));
#else
			is.setf(std::ios::failbit);
#endif
	}

#endif // !__NO_GENERIC_IOSTREAM

};

//===========================================================================
//!
//! \brief chromosome based on integers
//!
template < >
class ChromosomeT< int > : public ChromosomeT_num< int >
{
public:
	ChromosomeT()
	{ }
	explicit ChromosomeT< int >(unsigned l)
			: ChromosomeT_num< int >(l)
	{ }
	ChromosomeT(unsigned l, const int& v)
			: ChromosomeT_num< int >(l, v)
	{ }
	ChromosomeT(const std::vector< int >& v)
			: ChromosomeT_num< int >(v)
	{ }

	void mutateDiffGeom(double s);
	void mutateDiffGeom(const std::vector< double >& s,      bool = false);
	void mutateDiffGeom(const Chromosome& s,            bool = false);
	void mutateDiffGeom(const ChromosomeT< double >& s, bool = false);

protected:
	Chromosome* clone() const
	{
		return new ChromosomeT< int >(*this);
	}
	Chromosome* empty() const
	{
		return new ChromosomeT< int >;
	}


	/*! Part of PVM-send routine for type int Chromosomes */

	int pvm_pkchrom()
	{
		//cout << "\t  pk_chrom_int" << endl;
		unsigned i;

		unsigned *s = new unsigned;
		*s = this->size();
		pvm_pkuint(s, 1, 1);
		delete s;

		int *u = new int[this->size()];
		for (i = 0; i < this->size(); i++)
			u[i] = (*this)[i];
		pvm_pkint(u, this->size(), 1);
		delete[] u;

		return 1;
	};


	/*! Part of PVM-receive routine for type int Chromosomes */

	int pvm_upkchrom()
	{
		//cout << "\t  upk_chrom_int" << endl;
		unsigned i;

		unsigned *s = new unsigned;
		pvm_upkuint(s, 1, 1);
		(*this).resize(*s);
		delete s;

		int *u = new int[this->size()];
		pvm_upkint(u, this->size(), 1);
		for (i = 0; i < this->size(); i++)
			(*this)[i] = u[i];
		delete[] u;

		return 1;
	};


#ifndef __NO_GENERIC_IOSTREAM
	void writeTo(std::ostream& os) const
	{
		os << "ChromosomeT<" << typeid(int).name() << ">(" << size() << ")" << std::endl;
		for (unsigned i = 0; i < size(); i++)
		{
			if (i) os << '\t';
			os << (*this)[ i ];
		}
		os << std::endl;
	}

	void readFrom(std::istream& is)
	{
		std::string s;
		//is.getline( s );
		is >> s;
		is.get();    // skip end of line

		if (is.good() &&
				s.substr(0, 12) == "ChromosomeT<" &&
				s.find('>') != std::string::npos &&
				s.substr(12, s.find('>') - 12) == typeid(int).name())
		{

			resize(atoi(s.substr(s.find('>') + 2).c_str()));
			for (unsigned i = 0; i < size(); i++)
				is >> (*this)[ i ];
		}
		else
#if (defined(__GNUC__) && __GNUC__ > 2)
			// is.setf( std::ios::fmtflags(std::__ios_flags::_S_failbit) );
			is.setf(std::ios::fmtflags(std::ios::failbit));
#else
			is.setf(std::ios::failbit);
#endif
	}

#endif // !__NO_GENERIC_IOSTREAM

};

//===========================================================================
//!
//! \brief chromosome based on bits
//!
template < >
class ChromosomeT< bool > : public ChromosomeT_base< bool >
{
public:
	ChromosomeT()
	{ }
	explicit ChromosomeT(unsigned l)
			: ChromosomeT_base< bool >(l)
	{ }
	ChromosomeT(unsigned l, const bool& v)
			: ChromosomeT_base< bool >(l, v)
	{ }
	ChromosomeT(const std::vector< bool >& v)
			: ChromosomeT_base< bool >(v)
	{ }

	void   initialize();
	void   initialize(const unsigned pos);       // begin initialization at
	// index "pos"

	void   encode(double          val,
				  const Interval& range,
				  unsigned        nbits,
				  bool            useGray = false);
	double decode(const Interval& range,
				  bool useGray = false) const;

	void   encodeBinary(const std::vector< double >& chrom,
						const Interval&   range,
						unsigned          nbits,
						bool              useGray = false);

	void   encodeBinary(const Chromosome& chrom,
						const Interval&   range,
						unsigned          nbits,
						bool              useGray = false);

	void   flip(double p);
	void   flip(const std::vector< double >& p,
				bool cycle = false);
	void   flip(const Chromosome& p,
				bool cycle = false);

	bool operator == (const Chromosome& c) const;
	bool operator < (const Chromosome& c) const;

protected:
	Chromosome* clone() const
	{
		return new ChromosomeT< bool >(*this);
	}
	Chromosome* empty() const
	{
		return new ChromosomeT< bool >;
	}


	/*! Part of PVM-send routine for type bool Chromosomes */

	int pvm_pkchrom()
	{
		//cout << "\t  pk_chrom_bool" << endl;
		unsigned i;

		unsigned *s = new unsigned;
		*s = this->size();
		pvm_pkuint(s, 1, 1);
		delete s;

		unsigned *u = new unsigned[this->size()];
		for (i = 0; i < this->size(); i++)
			u[i] = (*this)[i];
		pvm_pkuint(u, this->size(), 1);
		delete[] u;

		return 1;
	};

	/*! Part of PVM-receive routine for type bool Chromosomes */

	int pvm_upkchrom()
	{
		//cout << "\t  upk_chrom_bool" << endl;
		unsigned i;

		unsigned *s = new unsigned;
		pvm_upkuint(s, 1, 1);
		(*this).resize(*s);
		delete s;

		unsigned *u = new unsigned[this->size()];
		pvm_upkuint(u, this->size(), 1);
		for (i = 0; i < this->size(); i++)
			(*this)[i] = (u[i] != 0);
		delete[] u;

		return 1;
	};

#ifndef __NO_GENERIC_IOSTREAM
	void writeTo(std::ostream& os) const
	{
		os << "ChromosomeT<" << typeid(bool).name() << ">(" << size() << ")" << std::endl;
		for (unsigned i = 0; i < size(); i++)
			os.put((*this)[ i ] ? '1' : '0');
		os << std::endl;
	}

	void readFrom(std::istream& is)
	{
		std::string   s;
		std::string::size_type pos = 0;

		//is.getline( s );

		is >> s;
		is.get();    // skip end of line

		if (is.good() &&
				s.substr(0, 12) == "ChromosomeT<" &&
				(pos = s.find('>')) != std::string::npos &&
				s.substr(12, pos - 12) == typeid(bool).name())
		{
			resize(atoi(s.substr(pos + 2).c_str()));
			char c;
			for (unsigned i = 0; i < size(); i++)
			{
				is.get(c);
				(*this)[ i ] = c != '0';
			}
		}
		else
#if (defined(__GNUC__) && __GNUC__ > 2)
			// is.setf( std::ios::fmtflags(std::__ios_flags::_S_failbit) );
			is.setf(std::ios::fmtflags(std::ios::failbit));
#else
			is.setf(std::ios::failbit);
#endif
	}

#endif // !__NO_GENERIC_IOSTREAM

};

//===========================================================================

#endif /* !__CHROMOSOMET_H */

