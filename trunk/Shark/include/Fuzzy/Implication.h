
/**
 * \file Implication.h
 *
 * \brief An implication
 * 
 * \authors Marc Nunkesser, Copyright (c) 2008, Marc Nunkesser
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

/* $log$ */

#ifndef IMPLICATION_H
#define IMPLICATION_H

#include <Fuzzy/FuzzySet.h>
#include <Fuzzy/FuzzyRelation.h>
#include <algorithm>
#include <Fuzzy/ComposedNDimFS.h>
#include <Fuzzy/RCPtr.h>

class NDimFS;

/**
 * \brief An implication.
 *
 * This class enables the user to configurate an implication by his own.
 *
 * <table border=1>
 *   <tr>
 *     <td>Zadeh</td>
 *     <td>\f$I_{Zad}(x,y)=max(min(x,y),1-x)\f$</td>
 *     <td>ZADEH</td>
 *   </tr>
 *   <tr>
 *     <td>Mamdani (Minimum)</td>
 *     <td>\f$I_{Mam}(x,y)=min(x,y)\f$</td>
 *     <td>MAMDANI</td>
 *   </tr>
 *   <tr>
 *     <td>Lukasiewicz</td>
 *     <td>\f$I_{Luk}(x,y)=min(1,1-x+y)\f$</td>
 *     <td>LUKASIEWICZ</td>
 *   </tr>
 *   <tr>
 *     <td>Goedel (Standard Star)</td>
 *     <td>\f$I_{Goed}(x,y)=\left\{\begin{array}{ll} 1 & \mbox{for } x \leq y \\ 
 *      y & \mbox{otherwise} \end{array}\right.\f$</td>
 *     <td>GOEDEL</td>
 *   </tr>
 *   <tr>
 *     <td>Kleene-Dienes</td>
 *     <td>\f$I_{Kle}(x,y)=max(1-x,y)\f$</td>
 *     <td>KLEENEDIENES</td>
 *   </tr>
 *   <tr>
 *     <td>Goguen (Gaines)</td>
 *     <td>\f$I_{Gog}(x,y)=\left\{\begin{array}{ll} 1 & \mbox{for } x=0\\
 *     min(1,y/x) & \mbox{otherwise}\end{array}\right.\f$</td>
 *     <td>GOGUEN</td>
 *   </tr>
 *   <tr>
 *     <td>Gaines-Reschner (Standard Strict)</td>
 *     <td>\f$I_{Gai}(x,y)=\left\{\begin{array}{ll} 1 & \mbox{for } x \leq y \\
 *     0 & \mbox{otherwise} \end{array}\right.\f$</td>
 *     <td>GAINESRESCHER</td>
 *   </tr>
 *   <tr>
 *     <td>Reichenbach(algebraic implication)</td>
 *     <td>\f$I_{Rei}(x,y)=1-x+xy\f$</td>
 *     <td>REICHENBACH</td>
 *   </tr>
 *   <tr>
 *     <td>Larsen</td>
 *     <td>\f$I_{Lar}(x,y)=xy\f$</td>
 *     <td>LARSEN</td>
 *   </tr>
 * </table> 
 * 
 */
class Implication:public FuzzyRelation {
public:

	enum ImplicationType {
		ZADEH,
		MAMDANI,
		LUKASIEWICZ,
		GOEDEL,
		KLEENEDIENES,
		GOGUEN,
		GAINESRESCHER,
		REICHENBACH,
		LARSEN
	};

/**
 * \brief Constructor.
 *
 * @param NDim1 first n-dimensional fuzzy set
 * @param NDim2 second n-dimensional fuzzy set
 * @param it the type of implication
 */ 
	Implication(const RCPtr<NDimFS>& NDim1,
	            const RCPtr<NDimFS>& NDim2,
	            ImplicationType it);

/**
 * \brief Destructor.
 */
	virtual ~Implication();

// overloaded operator():
	
	/**
	 * \brief Calculates the value of the implication \f$R(x,y)\f$ for the given 
	 * points.
	 * 
	 * @param x \f$x\f$
	 * @param y \f$y\f$
	 */
	virtual double operator()(const std::vector<double>&x,const std::vector<double>&y) const;
	
	/**
	 * \brief Calculates the implication given \f$x\f$ and \f$y\f$ left 
	 * variable.
	 * 
	 * @param x \f$x\f$
	 * @param lambda use: Lambda::Y
	 * 
	 * @return the implication
	 */
	virtual RCPtr<ComposedNDimFS> operator()(const std::vector<double>& x, Lambda lambda) const;
	
	/**
	 * \brief Returns 0.
	 */
	virtual RCPtr<ComposedNDimFS> operator()(Lambda, const std::vector<double>&) const;

private:
	static double                 Zadeh( double x, double y );
	static double                 Mamdani( double x, double y );
	static double                 Lukasiewicz( double x, double y );
	static double                 Goedel( double x, double y );
	static double                 KleeneDienes( double x, double y );
	static double                 Goguen( double x, double y);
	static double                 GainesRescher( double x, double y );
	static double                 Reichenbach( double x, double y );
	static double                 Larsen( double x, double y );

	MuType*                 usedFunction;
// pointer to "X-Fuzzyset" and "Y-FuzzySet" used for the parameters of the implication functions.
	const RCPtr<NDimFS>             xfs;
	const RCPtr<NDimFS>             yfs;
};

#endif
