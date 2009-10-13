//===========================================================================
/*!
 *  \file whitening.cpp
 *
 *  \brief Example for data whitening
 *
 *  \author  Björn Weghenkel
 *  \date    2009
 *
 *  \par Copyright (c) 1999-2009:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
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


#include <Array/ArrayIo.h>
#include <ReClaM/PCA.h>
#include <Rng/GlobalRng.h>

using namespace std;

int main( int argc, char**argv ) {
    
    // data arrays
    Array<double> data(1000, 2);
    Array<double> dataWhite;

    // output streams
    ofstream osPre("whitening_in");
    ofstream osPost("whitening_out");
    
    // generate correlated data
    for(unsigned int i = 0; i < data.dim(0); i++) {
        double x = Rng::gauss(2, 2);
        double y = Rng::gauss(-1, 4);
        data(i, 0) = 2 * x + y;
        data(i, 1) = x + 2 * y - .3;
    }
    
    // write correlated data to file
    writeArray( data, osPre );
    
    // model for whitening
    AffineLinearMap model( 2, 2 );
    PCA pca;
    pca.init( true );   // whitening
    pca.optimize( model, data );
    
    // whiten data and write to file
    model.model( data, dataWhite );
    writeArray( dataWhite, osPost );
    
    cout << "Wrote data to 'whitening_in' and 'whitening_out'." << endl;
    cout << "You can plot it with gnuplot:" << endl;
    cout << "  plot \"whitening_in\"" << endl;
    cout << "  plot \"whitening_out\"" << endl;
    
    return 0;
}
