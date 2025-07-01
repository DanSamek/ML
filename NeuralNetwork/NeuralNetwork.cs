using System.Collections.Concurrent;
using ML.NeuralNetwork.Loader;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

public partial class NeuralNetwork
{
    private readonly List<Layer> _layers = null!;
    private InputLayer _inputLayer = null!;
    private Func<ForwardResult, double> LossFunction { get; set; }
    private DataLoader _dataLoader = null!;

    private readonly ManualResetEvent _notEmptyQueueEvent = new (false);
    private readonly ManualResetEvent _emptyQueueEvent = new (false);
    private readonly ConcurrentQueue<TrainingItem?> _queue = new();
    
    private int _waitingWorkers;
    private double _trainingLoss = 0.0;
    
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
        LossFunction = lossFunction ?? throw new Exception("Loss function is not set.");
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
    /// </summary>
    /// <param name="path">Path where it should be saved.</param>
    public void Save(string path)
    {
        // TODO.
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
            var enumerator = _dataLoader.GetEnumerator();
            var total = 0;
           
            while (total < totalLines)
            {
                _trainingLoss = 0;
                // Load batch & Process batch.
                foreach (var (input, expected) in enumerator)
                {
                    if (total >= trainingOptions.BatchSize)
                        break;
                    
                    _queue.Enqueue(new TrainingItem(input, expected));
                    _notEmptyQueueEvent.Set();
                    
                    total++;
                }

                while (!_queue.IsEmpty)
                    _emptyQueueEvent.WaitOne();
                
                Console.WriteLine(_trainingLoss);

                // Backpropagation.
                // TODO
                
                foreach (var worker in workers)
                {
                    worker.UpdateNetwork();
                }
            }
        }

        // Stop workers.
        for (var i = 0; i < trainingOptions.NumberOfThreads; i++)
        {
            _queue.Enqueue(null);
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
            workers.Add(new Worker(this));
            
            var thread = new Thread(() => workers[^1].Run(numberOfThreads));
            threads.Add(thread);
        }
        
        foreach (var thread in threads)
            thread.Start();
        
        return (threads, workers);
    }
}