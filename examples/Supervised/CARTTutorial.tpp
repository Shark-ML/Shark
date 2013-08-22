//===========================================================================
/*!
 *  \brief CART Tutorial Sample Code
 *
 *  This file is part of the "CART" tutorial.
 *  It requires some toy sample data that comes with the library.
 *
 *  \author K. N. Hansen
 *  \date 2012
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

//###begin<includes>
#include <shark/Data/Csv.h> //importing csv files
#include <shark/Algorithms/Trainers/CARTTrainer.h> //the CART trainer
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> //zero one loss for evaluation
//###end<includes>
#include <iostream> 

using namespace std; 
using namespace shark;


int main() {

	//*****************LOAD AND PREPARE DATA***********************//
	
	//###begin<load>
	// read data
	ClassificationDataset dataTrain;
        import_csv(dataTrain, "data/C.csv", LAST_COLUMN, 0);


	//Split the dataset into a training and a test dataset
	ClassificationDataset dataTest =splitAtElement(dataTrain,311);
	
	cout << "Training set - number of data points: " << dataTrain.numberOfElements()
		 << " number of classes: " << numberOfClasses(dataTrain)
		 << " input dimension: " << inputDimension(dataTrain) << endl;

	cout << "Test set - number of data points: " << dataTest.numberOfElements()
		 << " number of classes: " << numberOfClasses(dataTest)
		 << " input dimension: " << inputDimension(dataTest) << endl;

	//###end<load>
	
	//Train the model
	//###begin<train>
	CARTTrainer trainer;
	CARTClassifier<RealVector> model;
	trainer.train(model, dataTrain);
	//###end<train>
	
	// evaluate Random Forest classifier
	//###begin<evaluation>
	ZeroOneLoss<unsigned int, RealVector> loss;
	Data<RealVector> prediction = model(dataTrain.inputs());
	cout << "CART on training set accuracy: " << 1. - loss.eval(dataTrain.labels(), prediction) << endl;

	prediction = model(dataTest.inputs());
	cout << "CART on test set accuracy:     " << 1. - loss.eval(dataTest.labels(), prediction) << endl;
	//###end<evaluation>

}
