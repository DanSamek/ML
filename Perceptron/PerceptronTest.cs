using NUnit.Framework;
using static ML.Perceptron.Perceptron;

namespace ML.Perceptron;

[TestFixture]
public class PerceptronTest
{   
    private const string DATA_SOURCE_PATH = "TODO"; 
    private const string LABEL_SOURCE_PATH = "TODO";

    private const int NUMBER_TO_TRAIN = 7;
    private static List<DatasetItem> LoadDataset()
    {
        // TODO find normal dataset.
        var result = new List<DatasetItem>();
        return result;
    }

    [Test]
    public void ImageTest()
    {
        var dataset = LoadDataset();
        var perceptron = new Perceptron(dataset);
        perceptron.Train();
        
        Assert.Pass();
    }
}