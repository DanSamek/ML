using System.Collections.Concurrent;
using ML.NeuralNetwork.Loader;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

public class NeuralNetwork
{
    public record TrainingItem(List<double> Input, List<double> Expected);
    public record ForwardResult(List<double> Output, List<double> Expected);
    
    
    private List<Layer> _layers;
    private InputLayer _inputLayer;
    private Func<ForwardResult, double> _lossFunction;
    private DataLoader _dataLoader;
    
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
    /// Runs learning of the neural network.
    /// </summary>
    public void Train(TrainingOptions trainingOptions)
    {
        var workerQueue = new ManualResetEvent(false);
        var mainThread = new ManualResetEvent(false);
        var queue = new ConcurrentQueue<TrainingItem?>();
        var waitingWorkers = 0;
        
        var workers = new List<Worker>();
        var threads = new List<Thread>();
        for (var i = 0; i < trainingOptions.NumberOfThreads; i++)
        {
            var worker = new Worker(_lossFunction, trainingOptions.NumberOfThreads);
            worker.UpdateWeights(_inputLayer, _layers);
            workers.Add(worker);
            
            var thread = new Thread(() => workers[^1].Run(ref workerQueue, ref mainThread, ref queue, ref waitingWorkers));
            threads.Add(thread);
        }
        
        foreach (var thread in threads)
            thread.Start();
        
        for (var epoch = 1; epoch <= trainingOptions.NumEpochs; epoch++)
        {
            var enumerator = _dataLoader.LoadItem();
            var total = 0;
            
            // Load batch & Process batch.
            foreach (var (input, expected) in enumerator)
            {
                if (total >= trainingOptions.BatchSize)
                    break;
                
                queue.Enqueue(new TrainingItem(input, expected));
                workerQueue.Set();
                
                total++;
            }

            while (!queue.IsEmpty)
                mainThread.WaitOne();

            // Backpropagation.
            // TODO
            
            // Update net in workers.
            foreach (var worker in workers)
            {
                worker.UpdateWeights(_inputLayer, _layers);
            }
        }

        // Stop workers.
        for (var i = 0; i < trainingOptions.NumberOfThreads; i++)
        {
            queue.Enqueue(null);
        }
        
        foreach (var thread in threads)
        {
            thread.Join();
        }
    }

    
    
    private class Worker
    {
        private List<Layer> _layers;
        private InputLayer _inputLayer;
        private readonly Func<ForwardResult, double> _lossFunction;
        private readonly int _totalWorkers;

        public Worker(Func<ForwardResult, double> lossFunction, int totalWorkers)
        {
            _lossFunction = lossFunction;
            _totalWorkers = totalWorkers;
        }

        public void UpdateWeights(InputLayer inputLayer, List<Layer> layers)
        {
            _inputLayer = inputLayer;
            _layers = layers;
        }
        
        public void Run(ref ManualResetEvent resetEvent, 
                        ref ManualResetEvent mainThread,
                        ref ConcurrentQueue<TrainingItem?> queue, 
                        ref int waitingWorkers)
        {
            while (true)
            {
                if (!queue.TryDequeue(out var item))
                {
                    Interlocked.Increment(ref waitingWorkers);
                    if (waitingWorkers >= _totalWorkers)
                    {
                        mainThread.Set();
                    }
                    
                    resetEvent.WaitOne();
                    Interlocked.Decrement(ref waitingWorkers);
                    continue;
                }
                
                // null = stop
                if (item is null)
                    break;
                
                Forward(item.Input);

                var output = _layers[^1].Neurons.Select(x => x.Sum).ToList();
                var forwardResult = new ForwardResult(output, item.Expected);
                var loss = _lossFunction(forwardResult);
                // TODO count forwarded + total loss.
            }
        }
        
        private void Forward(List<double> data)
        {
            ResetNeuronSums();
        
            for (var i = 0; i < _inputLayer.Size(); i++)
            {
                var weights = _inputLayer.Features[i].Weights;
                for (var j = 0; j < _layers[0].Size(); j++)
                    _layers[0].Neurons[j].Sum += data[i] * weights[j];
            }
        
            _layers[0].Activate();
            for (var layer = 0; layer < _layers.Count - 1; layer++)
            {
                var neurons = _layers[layer].Neurons;

                for (var i = 0; i < neurons.Count; i++)
                {
                    var weights = _layers[layer].Neurons[i].Weights;
                    for (var j = 0; j < _layers[layer + 1].Size(); j++)
                    {
                        _layers[layer + 1].Neurons[j].Sum += neurons[i].Sum * weights[j];
                    }
                }
                _layers[layer].Activate();
            }
            _layers[^1].Activate();
        }
        
        private void ResetNeuronSums() => _layers.ForEach(l => l.Neurons.ForEach(n => n.Sum = n.Bias));
    }
}