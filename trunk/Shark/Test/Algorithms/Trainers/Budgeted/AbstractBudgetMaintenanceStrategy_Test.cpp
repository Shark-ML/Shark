//===========================================================================
/*!
 *
 *
 * \brief       AbstractBudgetMaintenanceStrategy Test
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

#define BOOST_TEST_MODULE ABSTRACTBUDGETMAINTENANCESTRATEGY

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/Budgeted/AbstractBudgetMaintenanceStrategy.h>
#include <shark/Data/DataDistribution.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>


using namespace shark;

/*
// REMOVE ME           
            static void dumpModel(ModelType const& model) {
                for (size_t i = 0; i < model.basis().numberOfElements(); i++) {
                    for (size_t j = 0; j < model.alpha().size2(); j++) {
                        std::cout << (model.alpha(i, j)) << ", " ; 
                    }
                    std::cout   << "   -   ";
                    for (size_t j = 0; j < model.basis().element(j).size(); j++) {
                        std::cout << model.basis().element(i)(j) << ", " ; 
                    }
                    std::cout << "\n";
                }
            }
*/

BOOST_AUTO_TEST_CASE( AbstractBudgetMaintenanceStrategy_findSmallestVector)
{
    typedef RealVector InputType;
    typedef KernelExpansion<InputType> ModelType;
    
    
    // create a kernel
    double gamma = 1.0f;
    GaussianRbfKernel<> kernel(gamma);
    
    // create a fake 2d dataset with 64 entries and no noise
    size_t datasetSize = 64;
    Chessboard problem (2,0);
    LabeledData<RealVector, unsigned int> dataset = problem.generateDataset(datasetSize);
    unsigned int classes = numberOfClasses (dataset);
    
    // create a budget with some fake entries
    size_t m_budgetSize = 16;
    LabeledData<InputType, unsigned int> preinitializedBudgetVectors (m_budgetSize, dataset.element (0));
    
    // initialize kernel expansion 
    KernelClassifier<RealVector> classifier;
    ModelType &budgetModel = classifier.decisionFunction();

    // before we start: what happens if there is no vector in the budget?
    size_t index = 0;
    double minAlpha = -0.0;
    try {
        AbstractBudgetMaintenanceStrategy<RealVector>::findSmallestVector(budgetModel, index, minAlpha);
        
        // in release we must have infinity back
        BOOST_REQUIRE_EQUAL(minAlpha, std::numeric_limits<double>::infinity());
    } catch (...)
    {
        // in debug we got an exception, that is OK
        BOOST_REQUIRE_EQUAL(0, 0);
    }
    
    budgetModel.setStructure (&kernel, preinitializedBudgetVectors.inputs(), false, classes);
    
    // create the alphas, we need by just enumerating them
    for (size_t i = 0; i < m_budgetSize; i++)
        for (size_t j = 0; j < classes; j++)
            budgetModel.alpha(i, j) = i*classes + j;
    
    // find the smallest vector
    AbstractBudgetMaintenanceStrategy<RealVector>::findSmallestVector(budgetModel, index, minAlpha);
    BOOST_REQUIRE_EQUAL(index, 0);
    BOOST_REQUIRE_EQUAL(minAlpha, 1);
    
    // what happens if we have all alphas the same?
    for (size_t i = 0; i < m_budgetSize; i++)
        for (size_t j = 0; j < classes; j++)
            budgetModel.alpha(i, j) = classes;
    
    AbstractBudgetMaintenanceStrategy<RealVector>::findSmallestVector(budgetModel, index, minAlpha);
    BOOST_REQUIRE_EQUAL(index, 0);
    BOOST_REQUIRE_EQUAL(minAlpha, sqrt(2*classes*classes));
    
    // what happens if all things are zero in the budget?
    for (size_t i = 0; i < m_budgetSize; i++)
        for (size_t j = 0; j < classes; j++)
            budgetModel.alpha(i, j) = 0;
        
    AbstractBudgetMaintenanceStrategy<RealVector>::findSmallestVector(budgetModel, index, minAlpha);
    BOOST_REQUIRE_EQUAL(index, 0);
    BOOST_REQUIRE_EQUAL(minAlpha, 0);
}
	

	
	