/*!
*  \file CMAC.h
*
*  \author O. Krause
*
*  \brief implementation of a Cerebellar Model Articulation Controller (CMAC)
*  \par
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
*      ReClaM
*
*  <BR>
*
*
*  <BR><HR>
*  This file is part of ReClaM. This library is free software;
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
#ifndef CMAC_H
#define CMAC_H

#include<Array/Array.h>
#include<Array/ArrayOp.h>
#include<ReClaM/Model.h>
#include<Rng/GlobalRng.h>

//!
//! The CMACMap class represents a simple non-linear function
//! \f$ f : R^n \rightarrow R^m\f$
//!
class CMACMap : public Model
{
    private:
        /**
        *offset of the position of every tiling
        **/
        Array<double> m_offset;

        /**
        *coordinate offset for every dimension in the Array
        **/
        Array<size_t> m_dimOffset;

        /**
        *lower bound and tileWidth for every Dimension
        **/
        Array<double> m_tileBounds;

        /**
        *number of tilings
        **/
        size_t m_tilings;

        /**
        *number of parameters for each output dimension
        **/
        size_t m_parameters;
        /**
        *calculates the index in the parameter vector for the activated feature in the tiling
        **/
        size_t getArrayIndexForTiling(size_t indexOfTiling,Array<double> const& point)const;
    protected:
        /**
        *returns an index in the parameter array for each activated feature
        **/
        Array<size_t> getIndizes(Array<double> const& point)const;
    public:
        CMACMap();
        /**
        *construct the CMAC
        *
        *\param inputs number of input dimensions
        *\param outputs number of output dimensions
        *\param numberOfTilings number of Tilings to be created
        *\param numberOfTiles amount of tiles per dimension
        *\param lower lower bound of input values
        *\param upper upper bound of input values
        **/
        CMACMap(size_t inputs, size_t outputs, size_t numberOfTilings, size_t numberOfTiles, double lower = 0., double upper = 1.);
        /**
        *construct the CMAC
        *
        *\param inputs number of input dimensions
        *\param outputs number of output dimensions
        *\param numberOfTilings number of Tilings to be created
        *\param numberOfTiles amount of tiles per dimension
        *\param bounds lower and upper bounts for every input dimension
        **/
        CMACMap(size_t inputs, size_t outputs, size_t numberOfTilings, size_t numberOfTiles, Array<double> const& bounds);
        /**
        *initialize internal variables
        *
        *\param randomTiles if this is true, tilings will have a random offset
        **/
        void init(bool randomTiles=false);

        void model(Array<double> const& input, Array<double>& output);
        void modelDerivative(Array<double> const& input, Array<double>& derivative);

        /**
        *returns the mapped feature vector
        **/
        Array<double> getFeatureVector(Array<double> const& point)const;
};

//!
//! The CMACFunction class represents a simple non-linear function
//! \f$ f : R^n \rightarrow R\f$
//!
class CMACFunction : public CMACMap
{
    public:
        CMACFunction();
        /**
        *construct the CMAC
        *
        *\param inputs number of input dimensions
        *\param numberOfTilings number of Tilings to be created
        *\param numberOfTiles amount of tiles per dimension
        *\param lower lower bound of input values
        *\param upper upper bound of input values
        **/
        CMACFunction(size_t inputs, size_t numberOfTilings, size_t numberOfTiles, double lower = 0., double upper = 1.);
        /**
        *construct the CMAC
        *
        *\param inputs number of input dimensions
        *\param numberOfTilings number of Tilings to be created
        *\param numberOfTiles amount of tiles per dimension
        *\param bounds lower and upper bounts for every input dimension
        **/
        CMACFunction(size_t inputs, size_t numberOfTilings, size_t numberOfTiles, Array<double> const& bounds);
};
#endif

