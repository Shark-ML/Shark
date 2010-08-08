#include <ReClaM/CMAC.h>

size_t CMACMap::getArrayIndexForTiling(size_t indexOfTiling,Array<double> const& point)const
{

    size_t index = indexOfTiling * m_dimOffset(inputDimension);

    for(size_t dim = 0; dim != inputDimension; ++dim)
    {
        //Adjust range from (lower bound, upper bound) to (0,numberOfTiles)
        double coordinate = point(dim);
        coordinate -= m_tileBounds(dim, 0);//subtract lower bound
        coordinate -= m_offset(indexOfTiling, dim);//tiling offset
        //divide by the width of the tile to calculate the index
        coordinate /= m_tileBounds(dim, 1);
        //add index offset
        index += static_cast<size_t>(coordinate) * m_dimOffset(dim);
    }
    return index;
}

Array<size_t> CMACMap::getIndizes(Array<double> const& point)const
{
    Array<size_t> output(m_tilings);
    output=0.0;

    for(size_t tiling = 0; tiling != m_tilings; ++tiling)
    {
        size_t index = getArrayIndexForTiling(tiling, point);
        output(tiling) = index;
    }
    return output;
}

CMACMap::CMACMap():m_tilings(0){}

CMACMap::CMACMap(size_t inputs, size_t outputs, size_t numberOfTilings, size_t numberOfTiles, double lower, double upper)
    :m_offset(numberOfTilings, inputs), m_dimOffset(inputs + 1), m_tileBounds(inputs, 2), m_tilings(numberOfTilings)
{
    inputDimension  = inputs;
    outputDimension = outputs;
    m_tilings       = numberOfTilings;

    //initialize bounds
    //same for all dimensions
    double tileWidth = (upper - lower) / (numberOfTiles - 1);
    for(size_t dim = 0;dim != inputDimension; ++dim)
    {
        m_tileBounds(dim, 0) = lower;
        m_tileBounds(dim, 1) = tileWidth;
    }

    //calculate number of parameters and the offsets for every input dimension
    size_t numberOfParameters = 1;
    for(size_t inputDim = 0;inputDim != inputDimension; ++inputDim)
    {
        m_dimOffset(inputDim) = numberOfParameters;
        numberOfParameters *= numberOfTiles;
    }
    //parameters per tiling
    m_dimOffset(inputDimension) = numberOfParameters;

    //parameters for each output dimension
    numberOfParameters *= m_tilings;
    m_parameters=numberOfParameters;
    //parameters total
    numberOfParameters *= outputs;

    //initialize parameter array with random values
    parameter.resize(numberOfParameters);
    for(size_t i=0;i!=numberOfParameters;++i)
        parameter(i) = Rng::gauss(0,1);
}

CMACMap::CMACMap(size_t inputs, size_t outputs, size_t numberOfTilings, size_t numberOfTiles, Array<double> const& bounds)
    :m_offset(numberOfTilings, inputs), m_dimOffset(inputs + 1), m_tileBounds(inputs, 2), m_tilings(numberOfTilings)
{
    inputDimension  = inputs;
    outputDimension = outputs;
    m_tilings       = numberOfTilings;

    //initialize bounds
    for(size_t dim=0; dim != inputDimension; ++dim)
    {
        double tileWidth = (bounds(dim, 1) - bounds(dim, 0)) / (numberOfTiles - 1);
        m_tileBounds(dim, 0) = bounds(dim, 0);
        m_tileBounds(dim, 1) = tileWidth;
    }

    //calculate number of parameters and the offsets for every input dimension
    size_t numberOfParameters = 1;
    for(size_t inputDim = 0;inputDim != inputDimension; ++inputDim)
    {
        m_dimOffset(inputDim) = numberOfParameters;
        numberOfParameters *= numberOfTiles;
    }
    //parameters per tiling
    m_dimOffset(inputDimension) = numberOfParameters;

    //parameters for each output dimension
    numberOfParameters *= m_tilings;
    m_parameters=numberOfParameters;
    //parameters total
    numberOfParameters *= outputs;

    //initialize parameter array with random values
    parameter.resize(numberOfParameters);
    for(size_t i=0;i!=numberOfParameters;++i)
        parameter(i) = Rng::gauss(0,1);
}

void CMACMap::init(bool randomTiles)
{
    m_offset = 0;
    for(unsigned tiling = 0; tiling < m_tilings; ++tiling)
    {
        for(unsigned dim = 0; dim < inputDimension; ++dim)
        {
            if(!randomTiles)
                m_offset(tiling, dim) -= m_tileBounds(dim,1)/(tiling+1);
            else
                m_offset(tiling, dim) -= Rng::uni(0, m_tileBounds(dim,1));
        }
    }
}

void CMACMap::model(Array<double> const& input, Array<double>& output)
{
    if (input.ndim() == 1)
    {
        output.resize(outputDimension);
        output = 0;

        Array<size_t> indizes = getIndizes(input);
        for(size_t o=0;o!=outputDimension;++o)
        {
            for(size_t i = 0; i != m_tilings; ++i)
            {
                output(o) += parameter( indizes(i)+ o*m_parameters);
            }
        }
    }
    else if (input.ndim() == 2)
    {
        output.resize(input.dim(0), outputDimension,false);
        output = 0;

        for(size_t i = 0;i != input.dim(0); ++i)
        {
			Array<size_t> indizes = getIndizes(input[i]);
            for(size_t o=0;o!=outputDimension;++o)
            {
                for(size_t j = 0;j != m_tilings; ++j)
                {
                    output(i,o) += parameter( indizes(j) + o*m_parameters );
                }
            }
        }
    }
    else throw SHARKEXCEPTION("[CMAC::model] invalid number of dimensions.");
}

Array<double> CMACMap::getFeatureVector(Array<double> const& point)const
{
    Array<double> featureVector(m_parameters);
    featureVector = 0;

    Array<size_t> indizes = getIndizes(point);

    for(size_t j=0; j != m_tilings; ++j)
    {
        featureVector(indizes(j)) = 1;
    }

    return featureVector;
}

void CMACMap::modelDerivative(Array<double> const& input, Array<double>& derivative)
{
    if (input.ndim() != 1)
        throw SHARKEXCEPTION("[CMAC::modelDerivative] invalid number of dimensions.");

    derivative.resize(outputDimension,getParameterDimension(),false);
    derivative = 0;

    Array<size_t> indizes = getIndizes(input);

    for(size_t o=0;o!=outputDimension;++o)
    {
        for(size_t j=0; j != m_tilings; ++j)
        {
            derivative(o, indizes(j) + o*m_parameters ) = 1;
        }
    }
}

//CMACFunction


CMACFunction::CMACFunction(){}

CMACFunction::CMACFunction(size_t inputs, size_t numberOfTilings, size_t numberOfTiles, double lower, double upper)
    :CMACMap(inputs,1,numberOfTilings,numberOfTiles,lower,upper)
{}

CMACFunction::CMACFunction(size_t inputs, size_t numberOfTilings, size_t numberOfTiles, Array<double> const& bounds)
    :CMACMap(inputs,1,numberOfTilings,numberOfTiles,bounds)
{}
