using System.Collections.Concurrent;
using ML.NeuralNetwork.Loader;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

public partial class NeuralNetwork
{
    private Func<ForwardResult, double> _lossFunction;
    
    private readonly List<Layer> _layers = [];
    private InputLayer _inputLayer = null!;
    private DataLoader _dataLoader = null!;

    private readonly AutoResetEvent _notEmptyQueueEvent = new (false);
    private readonly AutoResetEvent _emptyQueueEvent = new (false);
    private readonly ConcurrentQueue<TrainingItem?> _queue = new();
    
    private int _waitingWorkers;
    private double _trainingLoss;
    
    /// <summary>
    /// Registers an input layer.
    /// </summary>
    /// <exception cref="Exception">If layer was already set.</exception>
    public NeuralNetwork AddInputLayer(int numberFeatures)
    {
        if (_inputLayer is not null)
        {
            throw new Exception("Input layer was already set.");
        }
        _inputLayer = new InputLayer(numberFeatures);
        return this;
    }
    
    /// <summary>
    /// Registers the hidden layer.
    /// </summary>
    /// <param name="numberOfNeurons">Number of neurons in layer.</param>
    /// <param name="activationFunction">Activation function that will be used in the layer.</param>
    /// <returns></returns>
    /// <exception cref="Exception">If activationFunction is not set.</exception>
    public NeuralNetwork AddHiddenLayer(int numberOfNeurons, Func<double, double> activationFunction)
    {
        if (activationFunction is null)
        {
            throw new Exception("Activation function was not set.");
        }
        
        var hiddenLayer = new Layer(numberOfNeurons, activationFunction);
        _layers.Add(hiddenLayer);
        return this;
    }
    
    /// <summary>
    /// Registers the loss function that will be used. 
    /// </summary>
    public NeuralNetwork SetLossFunction(Func<ForwardResult, double> lossFunction)
    {
        _lossFunction = lossFunction ?? throw new Exception("Loss function is not set.");
        return this;
    }
    
    /// <summary>
    /// "Connects" entire neural network.
    /// </summary>
    /// <returns></returns>
    /// <exception cref="Exception">If input layer is not set, or output layer is not set.</exception>
    public NeuralNetwork Build()
    {
        if (_inputLayer is null)  throw new Exception("Input layer is not set.");
        if (_layers.Count == 0)  throw new Exception("No layer has been set.");
        
        foreach (var feature in _inputLayer.Features)
            feature.Weights = InitListWithNItems<double>(_layers[0].Size());
        
        for (var i = 0; i < _layers.Count - 1; i++)
            foreach (var neuron in _layers[i].Neurons)
                neuron.Weights = InitListWithNItems<double>(_layers[i + 1].Size());
        
        return this;
    }

    /// <summary>
    /// Sets data loader that will be used.
    /// </summary>
    public NeuralNetwork SetDataLoader(DataLoader dataLoader)
    {
        _dataLoader = dataLoader;
        return this;
    }
    
    /// <summary>
    /// Saves neural network into a file.
    /// Format is for a layer:
    ///     Input layer weights.
    ///     [for each neuron]
    ///     neuron bias, neuron weights.
    /// </summary>
    /// <param name="path">Path where it should be saved.</param>
    public void Save(string path)
    {
        using var binaryStream = new BinaryWriter(File.Open(path, FileMode.Create));
        foreach (var weight in _inputLayer.Features.SelectMany(feature => feature.Weights))
            binaryStream.Write(weight);
        
        foreach (var neuron in _layers.SelectMany(l => l.Neurons))
        {
            binaryStream.Write(neuron.Bias);
            foreach (var weight in neuron.Weights)
                binaryStream.Write(weight);
        }
        binaryStream.Close();
    }
    
    /// <summary>
    /// Loads neural network.
    /// For format <see cref="Save"/>
    /// </summary>
    /// <param name="path">Path of the file where are the weights.</param>
    public void Load(string path)
    {
        using var binaryStream = new BinaryReader(File.Open(path, FileMode.Open));
        foreach (var feature in _inputLayer.Features)
        {
            for (var i = 0; i < feature.Weights.Count; i++)
                feature.Weights[i] = binaryStream.ReadDouble();
        }
        
        foreach (var neuron in _layers.SelectMany(l => l.Neurons))
        {
            neuron.Bias = binaryStream.ReadDouble();
            for (var i = 0; i < neuron.Weights.Count; i++)
                neuron.Weights[i] = binaryStream.ReadDouble();
        }
        
        binaryStream.Close();
    }
    
    /// <summary>
    /// Initializes neural network with random weights.
    /// </summary>
    /// <param name="min">Minimum weight value.</param>
    /// <param name="max">Maximum weight value.</param>
    public void InitializeRandom(int min = -10, int max = 10)
    {
        foreach (var feature in _inputLayer.Features)
        {
            for (var i = 0; i < feature.Weights?.Count; i++)
                feature.Weights[i] = RandomDouble();
        }
        
        foreach (var neuron in _layers.SelectMany(l => l.Neurons))
        {
            neuron.Bias = RandomDouble();
            for (var i = 0; i < neuron.Weights.Count; i++)
                neuron.Weights[i] = RandomDouble();
        }
        
        return;
        double RandomDouble() => Random.Shared.NextDouble() * Random.Shared.Next(min, max);
    }
    
    /// <summary>
    /// If quantization will be used when it saves/loads neural network.
    /// </summary>
    public void UseQuantization()
    {
        // TODO.
    }
    
    /// <summary>
    /// Runs training of the neural network.
    /// </summary>
    public void Train(TrainingOptions trainingOptions)
    {
        var (threads, workers) = RunWorkers(trainingOptions.NumberOfThreads);
        var totalLines = _dataLoader.CountLines();
        
        for (var epoch = 1; epoch <= trainingOptions.NumEpochs; epoch++)
        {
            var total = 0;
            while (total < totalLines)
            {
                _trainingLoss = 0;
                var currentBatchSize = 0;
                // Load batch & Process batch.
                while (true)
                {
                    if (currentBatchSize >= trainingOptions.BatchSize)
                        break;

                    var item = _dataLoader.GetNext();
                    if (item is null)
                        break;
                    
                    _queue.Enqueue(item);
                    _notEmptyQueueEvent.Set();
                    
                    total++;
                    currentBatchSize++;
                }

                while (!_queue.IsEmpty)
                    _emptyQueueEvent.WaitOne();
                
                // Backpropagation.
                // TODO
                
                foreach (var worker in workers)
                    worker.UpdateNetwork();
            }
        }

        StopWorkers(threads);
    }

    private void StopWorkers(List<Thread> threads)
    {
        for (var i = 0; i < threads.Count; i++)
        {
            _queue.Enqueue(null);
            _notEmptyQueueEvent.Set();
        }

        foreach (var thread in threads)
        {
            thread.Join();
        }
    }

    private (List<Thread> threads, List<Worker> workers) RunWorkers(int numberOfThreads)
    {
        var threads = new List<Thread>();
        var workers = new List<Worker>();
        for (var i = 0; i < numberOfThreads; i++)
        {
            var worker = new Worker(this, i);
            workers.Add(worker);
            
            var thread = new Thread(() => worker.Run(numberOfThreads));
            threads.Add(thread);
        }
        
        foreach (var thread in threads)
            thread.Start();
        
        return (threads, workers);
    }
}