using System.Text;
using ML.NeuralNetwork.Loader;
using NUnit.Framework;
using NUnit.Framework.Legacy;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

[TestFixture]
public class NeuralNetworkTests
{
    private readonly IReadOnlyList<List<double>> _loadTestData = new List<List<double>>
    {
        new () {0.5,-0.5,0.4},
        new () {1,1.9,1.14},
        new () {2.1,1,2.5405},
        new () {5.4,3,5.4},
        new () {6,2.12,1},
        new () {1.46,3.45,-5},
        new () {5.1,5.2,5.5}
    };
    
    private static string CreateFile(IReadOnlyList<List<double>> data)
    {
        var fileName = Guid.NewGuid().ToString();
        var stringBuilder = new StringBuilder();
        var lineSb = new StringBuilder();
        
        foreach (var item in data)
        {
            lineSb.Clear();
            for (var i = 0; i < item.Count - 1; i++)
                lineSb.Append($"{item[i]} ");
            lineSb.Append(item[^1]);
            
            var line = lineSb.ToString();
            stringBuilder.AppendLine(line);
        }

        var toSave = stringBuilder.ToString();
        File.WriteAllText(fileName, toSave);
        return fileName;
    }
    
    private static TrainingItem Parse(string line, int inputSize)
    {
        var splitLine = line.Split(" ");
        
        var input = new List<double>();
        for (var i = 0; i < inputSize; i++)
            input.Add(double.Parse(splitLine[i]));
        
        List<double> expected = [double.Parse(splitLine[^1])];
        var trainingItem = new TrainingItem(input, expected);
        return trainingItem;
    }
    
    private const string NN_NAME = "test.bin";
    
    [Test]
    public void LoadTest()
    {
        var dataFile = CreateFile(_loadTestData);
        var nn = new NeuralNetwork()
            .AddInputLayer(2)
            .AddLayer(1, value => value)
            .SetLossFunction(result => Math.Abs(result.Expected[0] - result.Output[0]))
            .SetDataLoader(new DataLoader(dataFile, item => Parse(item, 2))) 
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
                Assert.That(Math.Abs(beforeLayer.Neurons[ni].Bias - layer.Neurons[ni].Bias) < double.Epsilon);
                CollectionAssert.AreEqual(beforeLayer.Neurons[ni].Weights, layer.Neurons[ni].Weights);
            }
        }
        
        // Cleanup
        File.Delete(dataFile);
        File.Delete(NN_NAME);
    }
    
    /// <summary>
    /// Tests on simple input if forward is correct.
    /// With x => x / 2 activation function.
    /// </summary>
    [Test]
    public void SimpleForwardTest()
    {
        var dataFile = CreateFile(new List<List<double>>
            {
                new() { 1, 2, 3, 16.7125 },
            });

        var inputLayer = new InputLayer(3)
        {
            Features =
            [
                new Feature { Weights = [2, 1] },
                new Feature { Weights = [3, 1.5] },
                new Feature { Weights = [2, 4] }
            ]
        };
        
        var hiddenLayer1 = new Layer (2, x => x / 2)
        {
            Neurons = 
            [
                new Neuron { Bias = 0, Weights = [2, 2.5, 0] },
                new Neuron { Bias = 1, Weights = [2, 1.5, 1] }
            ]
        };

        var hiddenLayer2 = new Layer(3, x => x / 2)
        {
            Neurons = 
            [
                new Neuron { Bias = 1, Weights = [0.3] },
                new Neuron { Bias = 2, Weights = [1] },
                new Neuron { Bias = 3, Weights = [2] }
            ]
        };

        var outputLayer = new Layer(1, x => x / 2)
        {
            Neurons =
            [
                new Neuron { Bias = 1 }
            ]
        };
        
        // 3 -> 2 -> 3 -> 1 net
        var nn = new NeuralNetwork()
                .AddInputLayer(inputLayer)
                .AddLayer(hiddenLayer1)
                .AddLayer(hiddenLayer2)
                .AddLayer(outputLayer)
                .SetLossFunction(result =>
                {
                    Assert.That(Math.Abs(result.Expected[0] - result.Output[0]) < double.Epsilon);
                    return 0;
                })
                .SetDataLoader(new DataLoader(dataFile, item => Parse(item, 3)));
            
        nn.Train(new TrainingOptions());
        File.Delete(dataFile);
    }
}