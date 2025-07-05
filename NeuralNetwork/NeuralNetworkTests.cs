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
    
    
    private const string NN_NAME = "test.bin";

    private class TestFunction2 : ActivationFunctionBase
    {
        public override double Value(double x) => x;
        public override double RandomWeight(double inWeightCount, double outWeightCount) => Random.Shared.NextDouble();
    }
    
    [Test]
    public void LoadTest()
    {
        var dataFile = NeuralNetworkTestBase.CreateFile(_loadTestData);
        var nn = new NeuralNetwork()
            .AddInputLayer(2)
            .AddLayer(1, typeof(TestFunction2))
            .SetLossFunction(typeof(MAE))
            .SetDataLoader(new DataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 2))) 
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

        public override double RandomWeight(double inWeightCount, double outWeightCount) => 0;
    }

    private class SimpleForwardReceiver : IOutputReceiver
    {
        public void TrainingLoss(double loss)
        {
            Assert.That(loss == 0);
        }

        public void ValidationLoss(double loss)
        {
            throw new NotImplementedException();
        }
    }
    
    
    /// <summary>
    /// Tests on simple input if forward is correct.
    /// With x => x / 2 activation function.
    /// </summary>
    [Test]
    public void SimpleForwardTest()
    {
        var dataFile = NeuralNetworkTestBase.CreateFile(new List<List<double>>
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
                .SetDataLoader(new DataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 3)))
                .SetOutputReceiver(new SimpleForwardReceiver());
            
        nn.Train(new TrainingOptions());
        File.Delete(dataFile);
    }
    
    
    /// <summary>
    /// Test if loss is somehow lower and lower.
    /// </summary>
    [Test]
    public void BackPropagationTest()
    {
        var dataFile = NeuralNetworkTestBase.CreateFile(new List<List<double>>
        {
            new() { 1, 2, 3, 9 },
            new() { 5, 5, 3, 1 },
            new() { 5, 1, 3, 5 },
            new() { 5, 1, 0, 10 },
            new() { 1, 0, 3, 5 },
            new() { 1, 5, 3, 3 },
            new() { 1, 2, 13, 7 },
            new() { 13, 2, 1, 0 },
            new() { 5, 0, 3, 3 },
            new() { 5, 9, 1, 3 },
            new() { 13, 0, 1, 4 },
            new() { 5, 9, 3, 6 },
            new() { 9, 9, 9, 5 },
            new() { 6, 2, 3, 3 },
            new() { 1, 1, 0, 7 },
            new() { 1, 5, 9, 8 },
            new() { 10, 5, 3, 6 },
            new() { 1, 5, 10, 7 },
            new() { 1, 6, 1, 2 },
            new() { 5, 0, 9, 9 },
        });
        
        // 3 -> 5 -> 3 -> 1 net
        var nn = new NeuralNetwork()
            .AddInputLayer(3)
            .AddLayer(3, typeof(Sigmoid))
            .AddLayer(2, typeof(RELU))
            .AddLayer(1, typeof(Identity))
            .SetLossFunction(typeof(MSE))
            .SetDataLoader(new DataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 3)))
            .SetOutputReceiver(new ConsoleReceiver())
            .Build();

        nn.InitializeRandom();
        var options = new TrainingOptions
        {
            LearningRate = 0.01,
            NumEpochs = 500,
            BatchSize = 5
        };
        
        nn.Train(options);
        File.Delete(dataFile);
    }
    
    /// <summary>
    /// Check run with validation tests.
    /// </summary>
    [TestCase(1)]
    [TestCase(4)]
    public void ValidationDatasetTests(int numberOfThreads)
    {
        var dataFile = NeuralNetworkTestBase.CreateFile(new List<List<double>>
        {
            new() { 1, 2, 3, 9 },
            new() { 5, 5, 3, 1 },
            new() { 5, 1, 3, 5 },
            new() { 5, 1, 0, 10 },
            new() { 1, 0, 3, 5 },
            new() { 1, 5, 3, 3 },
            new() { 1, 2, 13, 7 },
            new() { 13, 2, 1, 0 },
            new() { 5, 0, 3, 3 },
            new() { 5, 9, 1, 3 },
            new() { 13, 0, 1, 4 },
            new() { 5, 9, 3, 6 },
            new() { 9, 9, 9, 5 },
            new() { 6, 2, 3, 3 },
            new() { 1, 1, 0, 7 },
            new() { 1, 5, 9, 8 },
            new() { 10, 5, 3, 6 },
            new() { 1, 5, 10, 7 },
            new() { 1, 6, 1, 2 },
            new() { 5, 0, 9, 9 },
        });

        var validationDataFile = NeuralNetworkTestBase.CreateFile(new List<List<double>>
        {
            new() { 6, 0, 3, 1 },
            new() { 1, 5, 3, 2 },
            new() { 2, 5, 10, 3 },
        });
        
        // 3 -> 5 -> 3 -> 1 net
        var nn = new NeuralNetwork()
            .AddInputLayer(3)
            .AddLayer(3, typeof(Sigmoid))
            .AddLayer(2, typeof(RELU))
            .AddLayer(1, typeof(Identity))
            .SetLossFunction(typeof(MSE))
            .SetDataLoader(new DataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 3)))
            .SetValidationDataLoader(new DataLoader(validationDataFile, item => NeuralNetworkTestBase.Parse(item, 3)))
            .SetOutputReceiver(new ConsoleReceiver())
            .Build();

        nn.InitializeRandom();
        var options = new TrainingOptions
        {
            LearningRate = 0.01,
            NumEpochs = 500,
            BatchSize = 5,
            NumberOfThreads = numberOfThreads
        };
        
        nn.Train(options);
        File.Delete(dataFile);
    }
    
    
    [Test]
    public void XORTrainingTest()
    {
        var dataFile = NeuralNetworkTestBase.CreateFile(new List<List<double>>
        {
            new() { 0, 0, 0 },
            new() { 0, 1, 1 },
            new() { 1, 0, 1 },
            new() { 1, 1, 0 },
        });

        
        var nn = new NeuralNetwork()
            .AddInputLayer(2)
            .AddLayer(2, typeof(Tanh))
            .AddLayer(1, typeof(Tanh))
            .SetLossFunction(typeof(MSE))
            .SetDataLoader(new DataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 2)))
            .SetOutputReceiver(new ConsoleReceiver())
            .Build();

        nn.InitializeRandom();
        var options = new TrainingOptions
        {
            LearningRate = 0.1,
            NumEpochs = 10000,
            BatchSize = 4
        };
        
        nn.Train(options);
        File.Delete(dataFile);
    }
}