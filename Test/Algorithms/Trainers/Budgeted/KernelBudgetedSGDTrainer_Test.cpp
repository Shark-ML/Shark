//===========================================================================
/*!
 *
 *
 * \brief       KernelBudgetedSGDTrainer Test
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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

#define BOOST_TEST_MODULE KERNELBUDGETEDSGDTRAINER

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/Budgeted/MergeBudgetMaintenanceStrategy.h>
#include <shark/Algorithms/Trainers/Budgeted/KernelBudgetedSGDTrainer.h>

#include <shark/Data/DataDistribution.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/ObjectiveFunctions/Loss/HingeLoss.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>


using namespace shark;


BOOST_AUTO_TEST_CASE( KernelBudgetedSGDTrainer_train)
{
    // Create a Gaussian RBF Kernel 
    double gamma = 1.0f;
    GaussianRbfKernel<> *kernel = new GaussianRbfKernel<> (gamma);

    // We will use the usual hinge Loss
    HingeLoss *hingeLoss = new HingeLoss();

    // As the budget maintenance strategy we choose the merge strategy
    // We need explicitly to state that we are merging RealVectors.
    // For the time being this is the only type of objects we can merge.
    MergeBudgetMaintenanceStrategy<RealVector> *strategy = new MergeBudgetMaintenanceStrategy<RealVector>();

    // Parameters for the trainer:
    // Our budget shall have at most 64 vectors
    size_t budgetSize = 32;
    // We want to run 3 epochs
    size_t epochs = 3;
    
    // Initialize the KernelBudgetedSGDTrainer and set number of epochs.
    std::cout << "Creating KernelBudgetedSGDTrainer." << std::endl;
    double cost = 1.0f;
    KernelBudgetedSGDTrainer<RealVector> *kernelBudgetedSGDtrainer =
        new KernelBudgetedSGDTrainer<RealVector> (kernel, hingeLoss, cost, false, false, budgetSize, strategy );
    kernelBudgetedSGDtrainer -> setEpochs (epochs);

    // We want to train a normal Chessboard problem.
    size_t datasetSize = 1000;
    std::cout << "Creating Chessboard dataset problem with " << datasetSize << " points." << std::endl;
    Chessboard problem (4);
    LabeledData<RealVector, unsigned int> trainingData = problem.generateDataset(datasetSize);

    // Create classifier that will hold the final model
    KernelClassifier<RealVector> kernelClassifier;
    
     // Train
    std::cout << "Training the KernelBudgetedSGDTrainer on the problem with a budget of " << budgetSize << " and " << epochs << " Epochs." << std::endl;
    kernelBudgetedSGDtrainer ->train (kernelClassifier, trainingData);

    // Check the number of support vectors first, it should be equal to the budgetSize (but can be less)
    Data<RealVector> supportVectors = kernelClassifier.decisionFunction().basis();
    
    size_t nSupportVectors = supportVectors.numberOfElements();
    std::cout << "We have " << nSupportVectors << " support vectors in our model.\n";
    if (nSupportVectors > budgetSize)
        SHARKEXCEPTION ("Something has gone wrong. There are more support vectors in the budget than specified!");

    // Create another test problem with 500 points
    std::cout << "Creating test data set with 500 points.\n";
    size_t testDatasetSize = 500;
    Chessboard testProblem (4);
    LabeledData<RealVector, unsigned int> testData = testProblem.generateDataset(testDatasetSize);
    
    // Check the performance on the test set
    std::cout << "Computing the performance on the test dataset using 0-1 loss.\n";
    ZeroOneLoss<unsigned int> loss;
    Data<unsigned int> prediction = kernelClassifier (testData.inputs());
    double error_rate = loss (testData.labels(), prediction);

    // Report performance.
    std::cout << "Test error rate: " << error_rate << std::endl;
}
	
	