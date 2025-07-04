using System.Text;
using ML.NeuralNetwork.ActivationFunctions;
using ML.NeuralNetwork.Loader;
using ML.NeuralNetwork.LossFunctions;
using ML.NeuralNetwork.OutputReceiver;
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

    private class TestFunction2 : ActivationFunctionBase
    {
        public override double Value(double x) => x;
    }
    
    [Test]
    public void LoadTest()
    {
        var dataFile = CreateFile(_loadTestData);
        var nn = new NeuralNetwork()
            .AddInputLayer(2)
            .AddLayer(1, typeof(TestFunction2))
            .SetLossFunction(typeof(MAE))
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
            var newLayer = new Layer(layer.Size(), layer.ActivationFunction.GetType());
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

    private class TestActivationFunction : ActivationFunctionBase
    {
        public override double Value(double x) => x / 2;
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
        
        var hiddenLayer1 = new Layer (2, typeof(TestActivationFunction))
        {
            Neurons = 
            [
                new Neuron { Bias = 0, Weights = [2, 2.5, 0] },
                new Neuron { Bias = 1, Weights = [2, 1.5, 1] }
            ]
        };

        var hiddenLayer2 = new Layer(3, typeof(TestActivationFunction))
        {
            Neurons = 
            [
                new Neuron { Bias = 1, Weights = [0.3] },
                new Neuron { Bias = 2, Weights = [1] },
                new Neuron { Bias = 3, Weights = [2] }
            ]
        };

        var outputLayer = new Layer(1, typeof(TestActivationFunction))
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
                .SetLossFunction(typeof(MAE))
                .SetDataLoader(new DataLoader(dataFile, item => Parse(item, 3)));
            
        nn.Train(new TrainingOptions());
        File.Delete(dataFile);
    }
    
    
    /// <summary>
    /// Test if loss is somehow lower and lower.
    /// </summary>
    [Test]
    public void BackPropagationTest()
    {
        var dataFile = CreateFile(new List<List<double>>
        {
            new() { 1, 2, 3, 50 },
            new() { 5, 5, 3, -10 },
            new() { 5, 5, 3, -19 },
            new() { 5, 1, 3, 12 },
            new() { 1, 0, 3, 79 },
            new() { 1, 5, 3, 71 },
            new() { 1, 1, 3, 5 },
            new() { 1, 2, 13, 7 },
            new() { 13, 2, 1, 0 },
            new() { 5, 0, 3, 3 },
            new() { 5, 9, 1, -39 },
        });
        
        // 3 -> 5 -> 3 -> 1 net
        var nn = new NeuralNetwork()
            .AddInputLayer(3)
            .AddLayer(5, typeof(Sigmoid))
            .AddLayer(3, typeof(Sigmoid))
            .AddLayer(1, typeof(Identity))
            .SetLossFunction(typeof(MSE))
            .SetDataLoader(new DataLoader(dataFile, item => Parse(item, 3)))
            .SetOutputReceiver(new ConsoleReceiver())
            .Build();

        nn.InitializeRandom(-1.4,1.4);
        var options = new TrainingOptions
        {
            LearningRate = 0.01,
            NumEpochs = 500,
            BatchSize = 12
        };
        
        nn.Train(options);
        File.Delete(dataFile);
    }
}