//===========================================================================
/*!
 *  \file FisherLDA.cpp
 *
 *  \par Copyright (c) 1998-2010:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@ini.ruhr-uni-bochum.de<BR>
 *      www:   http://www.ini.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \author Bjoern Weghenkel
 *  \date 2010
 *
 *  \par Project:
 *      ReClaM
 *
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
//===========================================================================


#include <Rng/GlobalRng.h>
#include <ReClaM/LinearModel.h>
#include <ReClaM/FisherLDA.h>
#include <ReClaM/Dataset.h>
#include <Array/ArrayIo.h>

#include <stdio.h>

using namespace std;

int main(int argc, char** argv)
{
  // load the data
  unsigned nClasses = 3;
  unsigned nExamples = 150;
  Dataset dataset;
  dataset.CreateFromFile( "iris.data", (int) nExamples );

	// construct LDA model for dimension reduction
  unsigned dim = 2;
	AffineLinearMap model( 4, dim );
	FisherLDA optimizer;
	optimizer.init();

	// train the model
	printf( "FisherLDA training ..." ); fflush(stdout);
	optimizer.optimize( model, dataset.getTrainingData(), dataset.getTrainingTarget() );
	printf( " done.\n" );

	// transform the data, that is, reduce the dimension from 4 to 2
	Array<double> reduced( nExamples, dim );
	model.model( dataset.getTrainingData(), reduced );

	// write reduced data to different files per class
	for( unsigned c = 0; c < nClasses; c++ ) {
	  ofstream outfile;
	  char outfilename[20];
	  sprintf( outfilename, "fisher-lda-%d.csv", c );
	  cout << "writing output to '" << outfilename << "' ..." << endl;
	  outfile.open( outfilename );
	  for( unsigned i = 0; i < nExamples; i++ ) {
	    if( dataset.getTrainingTarget()(i) == c ) {
	      for( unsigned j = 0; j < dim; j++ ) {
	        outfile << reduced(i,j) << " ";
	      }
	      outfile << endl;
	    }
	  }
	  outfile.close();
	}

	cout << endl;
	cout << "Now you can plot the result with your favorite tool, e.g. gnuplot:" << endl;
	cout << "plot \"fisher-lda-0.csv\", \"fisher-lda-1.csv\", \"fisher-lda-2.csv\"" << endl;

	// lines below are for self-testing this example, please ignore
	if (fabs(reduced(0, 0) - 2.57824) <= 1e-5) exit(EXIT_SUCCESS);
	else exit(EXIT_FAILURE);
}
