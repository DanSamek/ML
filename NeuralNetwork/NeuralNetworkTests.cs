using System.Text;
using ML.NeuralNetwork.Loader;
using NUnit.Framework;
using NUnit.Framework.Legacy;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

[TestFixture]
public class NeuralNetworkTests
{
    private readonly IReadOnlyList<(double, double, double)> _data = new List<(double, double, double)>()
    {
        (0.5,-0.5,0.4),
        (1,1.9,1.14),
        (2.1,1,2.5405),
        (5.4,3,5.4),
        (6,2.12,1),
        (1.46,3.45,-5),
        (5.1,5.2,5.5)
    };
    
    private const string NN_NAME = "test.bin";
    
    private string CreateFile()
    {
        var fileName = Guid.NewGuid().ToString();
        var stringBuilder = new StringBuilder();
        foreach (var item in _data)
        {
            var line = $"{item.Item1} {item.Item2} {item.Item3}";
            stringBuilder.AppendLine(line);
        }

        var toSave = stringBuilder.ToString();
        File.WriteAllText(fileName, toSave);
        return fileName;
    }
    
    private static TrainingItem Parse(string line)
    {
        var splitLine = line.Split(" ");
        List<double> input = [double.Parse(splitLine[0]), double.Parse(splitLine[1])]; 
        List<double> expected = [double.Parse(splitLine[^1])]; 
        var trainingItem = new TrainingItem(input, expected);
        return trainingItem;
    }

    // Absolute error.
    private double LossFunction(ForwardResult result) => Math.Abs(result.Expected[0] - result.Output[0]);
    
    [Test]
    public void LoadTest()
    {
        var fileName = CreateFile();
        var nn = new NeuralNetwork()
            .AddInputLayer(2)
            .AddHiddenLayer(1, value => value)
            .SetLossFunction(LossFunction)
            .SetDataLoader(new DataLoader(fileName, Parse)) 
            .Build();
        
        nn.InitializeRandom();
        
        // Copy entire net.
        var inputLayer = new InputLayer(nn.InputLayer.Size());
        for (var i = 0; i < inputLayer.Size(); i++)
            inputLayer.Features[i].Weights = [.. nn.InputLayer.Features[i].Weights];
        
        var layers = new List<Layer>();
        foreach (var layer in nn.Layers)
        {
            var newLayer = new Layer(layer.Size(), layer.ActivationFunction);
            for (var i = 0; i < newLayer.Size(); i++)
            {
                newLayer.Neurons[i].Bias = layer.Neurons[i].Bias;
                newLayer.Neurons[i].Weights = [.. layer.Neurons[i].Weights];
            }
            layers.Add(newLayer);
        }
        
        // Save & load
        nn.Save(NN_NAME);
        nn.Load(NN_NAME);
        
        Assert.That(layers != nn.Layers);
        Assert.That(inputLayer != nn.InputLayer);
        
        // Validate
        for (var i = 0; i < nn.InputLayer.Size(); i++)
            CollectionAssert.AreEqual(nn.InputLayer.Features[i].Weights, inputLayer.Features[i].Weights);
        
        for (var i = 0; i < nn.Layers.Count; i++)
        {
            var beforeLayer = layers[i];
            var layer = nn.Layers[i];

            for (var ni = 0; ni < layer.Size(); ni++)
            {
                Assert.That(beforeLayer.Neurons[ni].Bias == layer.Neurons[ni].Bias);
                CollectionAssert.AreEqual(beforeLayer.Neurons[ni].Weights, layer.Neurons[ni].Weights);
            }
        }
        
        // Cleanup
        File.Delete(fileName);
        File.Delete(NN_NAME);
    }
    
    /// <summary>
    /// TODO!
    /// First needed load net from file -- custom weights.
    /// </summary>
    [Test]
    public void ForwardTest()
    {
        /*
        var fileName = CreateFile();
        var nn = new NeuralNetwork()
                    .AddInputLayer(2)
                    .AddHiddenLayer(1, value => value)
                    .SetLossFunction(LossFunction)
                    .SetDataLoader(new DataLoader(fileName, Parse)) 
                    .Build();

        var options = new TrainingOptions();
        nn.Train(options);
        File.Delete(fileName);
        */
    }
}