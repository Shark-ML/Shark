//===========================================================================
/*!
 *  \file FisherLDA.cpp
 *
 *  \brief Train a (affine) linear map using Principal Component Analysis (PCA)
 *
 *  \author  T. Glasmachers
 *  \date    2007
 *
 *  \par
 *      This implementation is based upon a class removed from
 *      the LinAlg package, written by M. Kreutz in 1998.
 *
 *  \par Copyright (c) 1999-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#include <ReClaM/FisherLDA.h>

#include <Array/ArrayIo.h>
#include <Array/ArrayOp.h>
#include <LinAlg/LinAlg.h>
#include <LinAlg/VecMat.h>

#include <vector>


using namespace std;


FisherLDA::FisherLDA()
{
	bWhitening = false;
}

FisherLDA::~FisherLDA()
{
}


void FisherLDA::init(Model& model)
{
	AffineLinearMap* alm = dynamic_cast<AffineLinearMap*>(&model);
	if (alm == NULL) throw SHARKEXCEPTION("[FisherLDA::init] the model for FisherLDA must be an AffineLinearMap");
	if (alm->getInputDimension() < alm->getOutputDimension()) throw SHARKEXCEPTION("[FisherLDA::init] FisherLDA can not increase the dimensionality of the data");

	bWhitening = false;
}

void FisherLDA::init(bool whitening)
{
	bWhitening = whitening;
}

double FisherLDA::optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target)
{
	AffineLinearMap* alm = dynamic_cast<AffineLinearMap*>(&model);
	if (alm == NULL) throw SHARKEXCEPTION("[FisherLDA::optimize] the model for FisherLDA must be an AffineLinearMap");
	return optimize(*alm, input, target);
}

double FisherLDA::optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target, Array<double> &eigenvalues, Array<double> &trans)
{
	AffineLinearMap* alm = dynamic_cast<AffineLinearMap*>(&model);
	if (alm == NULL) throw SHARKEXCEPTION("[FisherLDA::optimize] the model for FisherLDA must be an AffineLinearMap");
	return optimize(*alm, input, target, eigenvalues, trans);
}

double FisherLDA::optimize(AffineLinearMap& model, const Array<double>& input, const Array<double>& target)
{
	Array<double> eigenvalues, trans;
	return optimize(model, input, target, eigenvalues, trans);
}


double FisherLDA::optimize(AffineLinearMap& model, const Array<double>& input, const Array<double>& targets, Array<double> &eigenvalues, Array<double> &trans)
{
	int i, ic = model.getInputDimension();
	int o, oc = model.getOutputDimension();

	Vector mean;
	Matrix scatter;
	MeanAndScatter( model, input, targets, mean, scatter );

	Matrix h(scatter);

	trans.resize(scatter, false);
	eigenvalues.resize(ic, false);

	if (bWhitening)
	{
		rankDecomp(scatter, trans, h, eigenvalues);
	}
	else
	{
		eigensymm(scatter, trans, eigenvalues);
	}
        
        //eigenvalues = fabs( eigenvalues );
        //eigensort( trans, eigenvalues );
                
        // set parameters
	int p = 0;
	for (o=0; o<oc; o++)
	{
		for (i=0; i<ic; i++)
		{
			model.setParameter(p, trans(i, o));
			p++;
		}
	}
	for (o=0; o<oc; o++)
	{
		double value = 0.0;
		for (i=0; i<ic; i++) value -= trans(i, o) * mean(i);
		model.setParameter(p, value);
		p++;
	}

	return 0.0;
}

void FisherLDA::MeanAndScatter(AffineLinearMap& model, const Array<double>& input, const Array<double>& target, Vector& mean, Matrix& scatter)
{
    SIZE_CHECK( input.ndim()  == 2 );
    SIZE_CHECK( target.ndim() == 2 );   // targets are two-dimensional
    SIZE_CHECK( target.dim(1) == 1 );   // but de facto one-dimensional
        
    // size of data
    unsigned int ne = input.dim(0); // number of examples
    unsigned int nd = input.dim(1); // dimensionality of data
    
    // intermediate results
    vector<Vector*> M;   // mean
    vector<Matrix*> S;   // scatter
    vector<unsigned> counter;   // counter for examples per class
    
    
    //
    // calculate mean and scatter for every class
    //
    
    // for every example in set ...
    for( unsigned i = 0; i < ne; i++ ) {
        
        // class label
        unsigned int k = target(i,0);

        // allocate memory if necessary
        if( M.size() <= k ) {
            M.resize( k+1 );
            S.resize( k+1 );
            counter.resize( k+1, 0 );
        }
        if( M[k] == 0 ) {
            M[k] = new Vector( nd );
            S[k] = new Matrix( nd, nd );
            *M[k] = 0.0;
            *S[k] = 0.0;
        }
        
        // count example
        counter[k] += 1;
    
        // add example to mean vector
        //*M[k] += input.row(i);
        for( unsigned j = 0; j < nd; j++ )
            M[k]->operator()(j) += input(i,j);
        
        // add example to scatter matrix
        //*S[k] += outerProduct( input.row(i), input.row(i) );
        for( unsigned j1 = 0; j1 < nd; j1++ )
            for( unsigned j2 = 0; j2 < nd; j2++ )
                S[k]->operator()(j1,j2) += input(i,j1) * input(i,j2);
    }
    
    // number of classes
    unsigned int nk = M.size();
    
    // check output dimensions
    AffineLinearMap* alm = dynamic_cast<AffineLinearMap*>(&model);
    if( alm == NULL )
        throw SHARKEXCEPTION( "[FisherLDA::optimize] the model for FisherLDA must be an AffineLinearMap" );
    if( alm->getOutputDimension() >= nk ) {
        SHARKEXCEPTION("[FisherLDA::optimize] dimensionality of output must be smaller than the number of classes");
    }
    
    // for every class ...
    for( unsigned int k = 0; k < nk; k++ ) {
        
        // normalize mean vector
        *M[k] /= (double) counter[k];
        
        // make scatter mean free
        *S[k] -= (double)( counter[k] ) * ( *M[k] % *M[k] );

        cout << endl << "*** M" << k << " ***" << endl;
        writeArray( *M[k], std::cout );
        cout << endl << "*** S" << k << "0 ***" << endl;
        writeArray( *S[k], std::cout );
    }
    
    //
    // calculate global mean and final scatter
    //

    Matrix Sb( nd, nd ); // between-class scatter
    Matrix Sw( nd, nd ); // within-class scatter
    
    Sb = 0.0;
    Sw = 0.0;

    // calculate global mean
    
    mean.resize( nd, false );
    mean = 0.0;
    
    for( unsigned int k = 0; k < nk; k++ )
        mean += (double)( counter[k] ) * (*M[k]);
    
    mean /= (double) ne;
    
    // calculate between- and within-class scatters
    
    for( unsigned int k = 0; k < nk; k++ ) {
        Vector m = (*M[k]) - mean;
        Sb += (double)( counter[k] ) * ( m % m );
        Sw += *S[k];
    }
    
    // invert Sw
    Matrix SwInv;
    invertSymm( SwInv, Sw );
    //SwInv = invert( Sw );
    //g_inverseCholesky( Sw, SwInv );
    //g_inverseMoorePenrose( Sw, SwInv );
    
    //scatter.resize( nd, nd, false );
    scatter = SwInv * Sb;
    
    ofstream filestream;
    filestream.open("M.csv");
    cout << endl << "*** M ***" << endl;
    writeArray( mean, filestream );
    filestream.close();

    filestream.open("Sb.csv");
    cout << endl << "*** Sb ***" << endl;
    writeArray( Sb, filestream );
    filestream.close();

    filestream.open("Sw.csv");
    cout << endl << "*** Sw ***" << endl;
    writeArray( Sw, filestream );
    filestream.close();

    filestream.open("SwInv.csv");
    cout << endl << "*** SwInv ***" << endl;
    writeArray( SwInv, filestream );
    filestream.close();

    filestream.open("SwInvSw.csv");
    cout << endl << "*** SwInv * Sw ***" << endl;
    writeArray( SwInv * Sw, filestream );
    filestream.close();

    filestream.open("scatter.csv");
    cout << endl << "*** scatter ***" << endl;
    writeArray( scatter, filestream );
    filestream.close();
    
    // free allocated memory
    for( unsigned int k = 0; k < nk; k++ ) {
        delete M[k];
        delete S[k];
    }
    
    return;
}
