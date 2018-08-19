//===========================================================================
/*!
 * 
 *
 * \brief       Principal Component Analysis Tutorial Sample Code
 * 
 * This file is part of the "Principal Component Analysis" tutorial.
 * The tutorial requires that you download the Cambridge Face Database
 * from http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
 * and adjust the facedirectory path to the directory containing the faces
 * in PGM format.
 * 
 * You need the libraries boost_serialization, boost_system, and 
 * boost_filesystem for this example.
 * 
 * 
 *
 * \author      C. Igel
 * \date        2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

//###begin<includes>
#include <shark/Algorithms/Trainers/PCA.h>
#include <shark/Data/Pgm.h>
//###end<includes>

#include <fstream>
#include <boost/format.hpp>

using namespace std;
using namespace shark;


int main(){
	// read image data
	//###begin<import>
	const char *facedirectory = "Cambridge_FaceDB"; //< set this to the directory containing the face database
	Data<RealVector> images;
	//###end<import>
	cout << "Read images ... " << flush;
	try {
	//###begin<import>
		importPGMSet(facedirectory, images);
	//###end<import>
	} catch(...) {
		cerr << "[PCATutorial] could not open face database directory\n\nThis file is part of the \"Principal Component Analysis\" tutorial.\nThe tutorial requires that you download the Cambridge Face Database\nfrom http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html\nand adjust the facedirectory path in the source code to the directory\ncontaining the faces in PGM format." << endl;
		return 1;
	}
	cout << "done." << endl;
	
	//###begin<import>
	std::size_t l = images.numberOfElements();   // number of samples
	std::size_t x = images.shape()[1]; // width of images
	std::size_t y = images.shape()[0]; // height of images
	//###end<import>
	
	cout << "Eigenvalue decomposition ... " << flush;
	//###begin<pca>
	PCA pca(images);
	//###end<pca>
	cout << "done." << endl;
	
	cout << "Writing mean face and eigenvalues... " << flush;
	ofstream ofs("facesEigenvalues.csv");
	for(unsigned i=0; i<l; i++) 
		ofs << pca.eigenvalue(i) << endl;
	//###begin<export_mean>
	image::writeImageToFile<double>("facesMean.pgm", pca.mean(), {x, y, 1}, PixelType::Luma );
	//###end<export_mean>
	cout << "done. " << endl;

	cout << "Encoding ... " << flush;
	//###begin<model_encoder>
	unsigned m = 299;
	LinearModel<> enc;
	pca.encoder(enc, m);
	//###end<model_encoder>
	Data<RealVector> encodedImages = enc(images);
	cout << "done. " << endl;

	//###begin<model_reconstruction>
	unsigned sampleImage = 0;
	//###end<model_reconstruction>
	cout << "Reconstructing face " << sampleImage << " ... " << flush;
	boost::format fmterTrue("face%d.pgm");
	image::writeImageToFile<double>("face" + std::to_string(sampleImage) + ".pgm", elements(images)[sampleImage], {x, y, 1}, PixelType::Luma );
	//###begin<model_decoder>
	LinearModel<> dec;
	pca.decoder(dec, m);
	//###end<model_decoder>
	//###begin<model_reconstruction>
	std::string filename = "facesReconstruction" + std::to_string(sampleImage) + "-" + std::to_string(m) + ".pgm";
	auto reconstruction = dec(elements(encodedImages)[sampleImage]);
	image::writeImageToFile<double>(filename, reconstruction, {x, y, 1}, PixelType::Luma );
	//###end<model_reconstruction>
	cout << "done." << endl;
}
