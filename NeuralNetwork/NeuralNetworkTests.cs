using System.Text;
using ML.NeuralNetwork.ActivationFunctions;
using ML.NeuralNetwork.Loader;
using ML.NeuralNetwork.LossFunctions;
using ML.NeuralNetwork.Optimizers;
using ML.NeuralNetwork.OutputReceiver;
using NUnit.Framework;
using NUnit.Framework.Legacy;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

[TestFixture]
public class NeuralNetworkTests
{
    private readonly IReadOnlyList<List<double>> _IOTestData = new List<List<double>>
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

    private (InputLayer InputLayer, List<Layer> Layers) CopyNet(NeuralNetwork nn)
    {
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
        return (inputLayer, layers);
    }
    
    /// <summary>
    /// Load & save test.
    /// </summary>
    [Test]
    public void IOTest()
    {
        var dataFile = NeuralNetworkTestBase.CreateFile(_IOTestData);
        var nn = new NeuralNetwork()
            .AddInputLayer(2)
            .AddLayer(2, typeof(TestFunction2))
            .AddLayer(3, typeof(TestFunction2))
            .AddLayer(1, typeof(TestFunction2))
            .SetLossFunction(typeof(MAE))
            .SetDataLoader(new ShuffleDataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 2),2,1)) 
            .Build();
        
        nn.InitializeRandom();

        var (inputLayer, layers) = CopyNet(nn);
        
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

    
    private class IOQuantReceiver : IOutputReceiver
    {
        internal double LastTrainingLoss { get; private set; }
        internal  double LastValidationLoss { get; private set; }
        
        public void TrainingLoss(double loss)  => LastTrainingLoss = loss;

        public void ValidationLoss(double loss) => LastValidationLoss = loss;
        public void EpochCompleted(int epoch, double totalLoss) { }
        public void EpochCompleted(int epoch) { }
    }
    
    /// <summary>
    /// Load & save test with quantized weights.
    /// </summary>
    [Test]
    [Repeat(1000, true)]
    public void IOQuantizedTest()
    {
        var dataFile = NeuralNetworkTestBase.CreateFile(_IOTestData);
        var reciever = new IOQuantReceiver();
        var nn = new NeuralNetwork()
            .AddInputLayer(2)
            .AddLayer(2, typeof(TestFunction2))
            .AddLayer(3, typeof(TestFunction2))
            .AddLayer(1, typeof(TestFunction2))
            .SetLossFunction(typeof(MAE))
            .SetDataLoader(new ShuffleDataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 2),2,1))
            .SetValidationDataLoader(new ShuffleDataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 2),2,1))
            .SetOutputReceiver(reciever)
            .UseQuantization([8, 8, 8])
            .Build();
        
        nn.InitializeRandom();
        
        var (inputLayer, layers) = CopyNet(nn);
        
        nn.Save(NN_NAME);
        nn.Load(NN_NAME);
        
        Assert.That(layers != nn.Layers);
        Assert.That(inputLayer != nn.InputLayer);
        
        // Validate
        for (var i = 0; i < nn.InputLayer.Size(); i++){
            for (var j = 0; j < nn.InputLayer.Features[i].Weights.Count; j++)
            {
                Assert.That(double.Abs(nn.InputLayer.Features[i].Weights[j] - inputLayer.Features[i].Weights[j]) < 0.1);   
            }
        }
        
        for (var i = 0; i < nn.Layers.Count; i++)
        {
            var beforeLayer = layers[i];
            var layer = nn.Layers[i];

            for (var ni = 0; ni < layer.Size(); ni++)
            {
                Assert.That(Math.Abs(beforeLayer.Neurons[ni].Bias - layer.Neurons[ni].Bias) < 0.1);

                for (var j = 0; j < beforeLayer.Neurons[ni].Weights.Count; j++)
                {
                    Assert.That(double.Abs(beforeLayer.Neurons[ni].Weights[j] - layer.Neurons[ni].Weights[j]) < 0.1);
                }
            }
        }
        
        File.Delete(NN_NAME);
        File.Delete(dataFile);
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

        public void ValidationLoss(double loss) { }
        public void EpochCompleted(int epoch, double totalLoss) { }

        public void EpochCompleted(int epoch) { }
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
                .SetDataLoader(new ShuffleDataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 3),3,1))
                .SetOutputReceiver(new SimpleForwardReceiver());
            
        nn.Train(new TrainingOptions());
        File.Delete(dataFile);
    }

    
    /// <summary>
    /// Test if loss is somehow lower and lower.
    /// </summary>
    [Test]
    [TestCase(1)]
    [TestCase(2)]
    [TestCase(4)]
    [TestCase(8)]
    [TestCase(16)]
    public void BackPropagationTest(int numberOfThreads)
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
        
        // 3 -> 5 -> 3 -> 1 net
        var nn = new NeuralNetwork()
            .AddInputLayer(inputLayer)
            .AddLayer(hiddenLayer1)
            .AddLayer(hiddenLayer2)
            .AddLayer(outputLayer)
            .SetLossFunction(typeof(MSE))
            .SetDataLoader(new ShuffleDataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 3),3,1))
            .SetOutputReceiver(new ConsoleReceiver())
            .SetOptimizer(new Adam
            {
                Configuration = new Adam.Config()
            });

        var options = new TrainingOptions
        {
            NumEpochs = 200,
            BatchSize = 10,
            NumberOfThreads = numberOfThreads
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
            .SetDataLoader(new ShuffleDataLoader(dataFile, context => NeuralNetworkTestBase.Parse(context, 3), 3,1))
            .SetValidationDataLoader(new ShuffleDataLoader(validationDataFile, context => NeuralNetworkTestBase.Parse(context, 3), 3, 1))
            .SetOutputReceiver(new ConsoleReceiver())
            .Build();

        nn.InitializeRandom();
        var options = new TrainingOptions
        {
            NumEpochs = 500,
            BatchSize = 5,
            NumberOfThreads = numberOfThreads
        };
        
        nn.Train(options);
        File.Delete(dataFile);
    }

    
    private class MSE2 : LossFunctionBase
    {
        public double Value(double current, double expected) => Math.Pow(expected - current, 2) / 2.0;

        public double Derivative(double current, double expected) => expected - current;
    }
    
    [Test]
    public void XORTrainingTest2()
    {
        var dataFile = NeuralNetworkTestBase.CreateFile(new List<List<double>>
        {
            new() { 1, 1, 0 },
            new() { 0, 1, 1 },
            new() { 1, 0, 1 },
            new() { 0, 0, 0 }
        });
        

        var simple = new Simple
        {
            Configuration = new Simple.Config
            {
                LearningRate = 1
            }
        };
        
        var inputLayer = new InputLayer(2)
        {
            Features =
            [
                new Feature { Weights = [0.30006336, -0.64114065, 0.01708628 ] },
                new Feature { Weights = [-0.99758489, -0.37674861, -0.7799422] }
            ]
        };
        
        var hiddenLayer1 = new Layer (3, typeof(Sigmoid))
        {
            Neurons = 
            [
                new Neuron { Bias = 0, Weights = [-0.85391348] },
                new Neuron { Bias = 0, Weights = [ 0.04359306] },
                new Neuron { Bias = 0, Weights = [ -0.78181754] }
            ]
        };

        var outputLayer = new Layer(1, typeof(Sigmoid))
        {
            Neurons = 
            [
                new Neuron { Bias = 0, Weights =  [] },
            ]
        };

        var nn = new NeuralNetwork()
            .AddInputLayer(inputLayer)
            .AddLayer(hiddenLayer1)
            .AddLayer(outputLayer)
            .SetLossFunction(typeof(MSE2))
            .SetDataLoader(new DataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 2), 2, 1))
            .SetOutputReceiver(new ConsoleReceiver());
        
        var options = new TrainingOptions
        {
            NumEpochs = 1,
            BatchSize = 4
        };
        
        nn.Train(options);
        File.Delete(dataFile);
    }

    
    [Test]
    public void XORTrainingTest()
    {
        var dataFile = NeuralNetworkTestBase.CreateFile(new List<List<double>>
        {
            new() { 1, 1, 0 },
            new() { 0, 1, 1 },
            new() { 1, 0, 1 },
            new() { 0, 0, 0 }
        });
        
        var adam = new Adam
        {
            Configuration = new Adam.Config()
        };

        var nn = new NeuralNetwork()
            .AddInputLayer(2)
            .AddLayer(2, typeof(Tanh))
            .AddLayer(1, typeof(Sigmoid))
            .SetLossFunction(typeof(MSE))
            .SetDataLoader(new DataLoader(dataFile, item => NeuralNetworkTestBase.Parse(item, 2),2 ,1))
            .SetOutputReceiver(new ConsoleReceiver())
            .SetOptimizer(adam) 
            .Build();
        
        nn.InitializeRandom();
        var options = new TrainingOptions
        {
            NumEpochs = 10000,
            BatchSize = 4
        };
        
        nn.Train(options);
        File.Delete(dataFile);
    }
}