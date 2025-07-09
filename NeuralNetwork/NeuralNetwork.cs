using System.Collections.Concurrent;
using ML.NeuralNetwork.Loader;
using ML.NeuralNetwork.LossFunctions;
using ML.NeuralNetwork.Optimizers;
using ML.NeuralNetwork.OutputReceiver;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

public partial class NeuralNetwork
{
    public List<Layer> Layers { get; } = [];
    public InputLayer InputLayer { get; private set; } = null!;
    public Layer OutputLayer => Layers[^1];
    
    private LossFunctionBase _lossFunction = null!;
    private IOutputReceiver? _outputReceiver;
    private DataLoader _dataLoader = null!;
    private DataLoader? _validationDataLoader;
    private IOptimizer[] _optimizers = null!;
    private List<double> _scales = [];
    
    private readonly AutoResetEvent _notEmptyQueueEvent = new (false);
    private readonly AutoResetEvent _emptyQueueEvent = new (false);
    private readonly ConcurrentQueue<TrainingItem?> _queue = new();
    
    private IOptimizer _optimizer = new Simple
    {
        Configuration = new Simple.Config()
    };
    
    private int _waitingWorkers;
    private double _trainingLoss, _validationLoss;
    
    /// <summary>
    /// Sets the output receiver.
    /// </summary>
    public NeuralNetwork SetOutputReceiver(IOutputReceiver outputReceiver)
    {
        _outputReceiver = outputReceiver;
        return this;
    }
    
    /// <summary>
    /// Registers an input layer.
    /// </summary>
    /// <exception cref="Exception">If layer was already set.</exception>
    public NeuralNetwork AddInputLayer(int numberFeatures)
    {
        if (InputLayer is not null)  
            throw new Exception("Input layer was already set.");
        
        InputLayer = new InputLayer(numberFeatures);
        return this;
    }
    
    /// <summary>
    /// Sets input layer.
    /// </summary>
    /// <returns></returns>
    public NeuralNetwork AddInputLayer(InputLayer inputLayer)
    {
        if (InputLayer is not null)  
            throw new Exception("Input layer was already set.");
        
        InputLayer = inputLayer;
        return this;
    }
    
    /// <summary>
    /// Adds a new layer [hidden/output]
    /// </summary>
    /// <param name="numberOfNeurons">Number of neurons in layer.</param>
    /// <param name="activationFunction">Activation function that will be used in the layer.</param>
    /// <returns></returns>
    /// <exception cref="Exception">If activationFunction is not set.</exception>
    public NeuralNetwork AddLayer(int numberOfNeurons, Type activationFunction)
    {
        if (activationFunction is null)
        {
            throw new Exception("Activation function was not set.");
        }
        
        var hiddenLayer = new Layer(numberOfNeurons, activationFunction);
        Layers.Add(hiddenLayer);
        return this;
    }
    
    /// <summary>
    /// Adds a new layer [hidden/output]
    /// </summary>
    /// <param name="layer">Configured layer.</param> 
    /// <exception cref="Exception">If activationFunction is not set.</exception>
    /// <returns></returns>
    
    public NeuralNetwork AddLayer(Layer layer)
    {
        Layers.Add(layer);
        return this;
    }
    
    /// <summary>
    /// Registers the loss function that will be used. 
    /// </summary>
    public NeuralNetwork SetLossFunction(Type lossFunction)
    {
        _lossFunction = Activator.CreateInstance(lossFunction) as LossFunctionBase ?? throw new Exception("Incorrect type of the loss function.");
        return this;
    }
    
    /// <summary>
    /// "Connects" entire neural network.
    /// NOTE: Can be used only if <see cref="AddLayer(int,System.Type)"/> and <see cref="AddInputLayer(int)"/> was used.
    /// </summary>
    /// <returns></returns>
    /// <exception cref="Exception">If input layer is not set, or output layer is not set.</exception>
    public NeuralNetwork Build()
    {
        ArgumentNullException.ThrowIfNull(InputLayer);
        ArgumentOutOfRangeException.ThrowIfEqual(Layers.Count, 0);
        
        foreach (var feature in InputLayer.Features)
            feature.Weights = InitListWithNItems<double>(Layers[0].Size());
        
        for (var i = 0; i < Layers.Count - 1; i++)
            foreach (var neuron in Layers[i].Neurons)
                neuron.Weights = InitListWithNItems<double>(Layers[i + 1].Size());
        
        return this;
    }

    /// <summary>
    /// Sets data loader that will be used for training.
    /// </summary>
    public NeuralNetwork SetDataLoader(DataLoader dataLoader)
    {
        _dataLoader = dataLoader;
        return this;
    }
    
    /// <summary>
    /// Sets data loader that will be used for validation-error.
    /// </summary>
    public NeuralNetwork SetValidationDataLoader(DataLoader dataLoader)
    {
        _validationDataLoader = dataLoader;
        return this;
    }
    
    /// <summary>
    /// Sets optimizer that will be used.
    /// Default is set to <see cref="Simple"/>
    /// </summary>
    public NeuralNetwork SetOptimizer(IOptimizer optimizer)
    {
        _optimizer = optimizer;
        return this;
    }

    /// <summary>
    /// If quantization will be used when it saves/loads neural network.
    /// Weight quantization per layer is used.
    /// </summary>
    public NeuralNetwork UseQuantization(List<double> scales)
    {
        _scales = scales;
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
        var useQuantization = _scales.Count != 0;
        if (useQuantization) SaveQuantized(path);
        else SaveNormal(path);
    }

    /// <summary>
    /// Loads neural network.
    /// For format <see cref="Save"/>
    /// </summary>
    /// <param name="path">Path of the file where are the weights.</param>
    public void Load(string path)
    {
        var useQuantization = _scales.Count != 0;
        if (useQuantization) LoadQuantized(path);
        else LoadNormal(path);
    }
    
    /// <summary>
    /// Initializes neural network with random weights.
    /// </summary>
    public void InitializeRandom()
    {
        for (var i = 0; i < InputLayer.Features.Count; i++)
        {
            var feature = InputLayer.Features[i];
            for (var j = 0; j < feature.Weights.Count; j++)
                feature.Weights[j] = RandomDouble(0);
        }

        for (var i = 0; i < Layers.Count - 1; i++)
        {
            var neurons = Layers[i].Neurons;
            foreach (var neuron in neurons)
            {
                neuron.Bias = 0; // TODO

                for (var j = 0; j < neuron.Weights.Count; j++)
                    neuron.Weights[j] = RandomDouble(i + 1);
            }
        }
        
        return;

        double RandomDouble(int layerIdx)
        {
            var last = Layers.Count - 1 == layerIdx;
            var first = layerIdx == 0;
            
            var weightsIn = first ? InputLayer.Size() : Layers[layerIdx - 1].Size();
            var weightsOut = last ? 0 : Layers[layerIdx + 1].Size();
            
            var rw = Layers[layerIdx].ActivationFunction.RandomWeight(weightsIn, last ? 0 : weightsOut);
            return rw;
        }
    }
    
    /// <summary>
    /// Runs training of the neural network.
    /// </summary>
    public void Train(TrainingOptions trainingOptions)
    {
        ArgumentNullException.ThrowIfNull(_lossFunction);
        
        var (threads, workers) = RunWorkers(trainingOptions.NumberOfThreads);
        var totalLines = _dataLoader.CountLines();
        InitOptimizers();
        
        for (var epoch = 1; epoch <= trainingOptions.NumEpochs; epoch++)
        {
            _dataLoader.Reset();
            
            var total = 0;
            while (total < totalLines)
            {
                foreach (var worker in workers)
                    worker.ClearGradients();
                
                var (weightGradients, biasGradients) = CreateArraysForGradients(this);
                
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
                
                _outputReceiver?.TrainingLoss(_trainingLoss);
                
                if (_trainingLoss == 0)
                    continue;
                
                SumGradients(workers, weightGradients, biasGradients);
                AverageGradients(weightGradients, biasGradients, currentBatchSize);
                UpdateWeights(weightGradients, biasGradients);
            }
            
            CalculateValidationLoss();
        }
        
        StopWorkers(threads);
    }

    private void CalculateValidationLoss()
    {
        if (_validationDataLoader is null)
            return;
        
        _validationDataLoader?.Reset();
        _validationLoss = 0;
        
        while (true)
        {
            var item = _validationDataLoader?.GetNext();
            if (item is null)
                break;
            
            item = item with
            {
                Validation = true
            };
            _queue.Enqueue(item);
            _notEmptyQueueEvent.Set();
        }
        
        while (!_queue.IsEmpty)
            _emptyQueueEvent.WaitOne();
        
        _outputReceiver?.ValidationLoss(_validationLoss);
    }
    
    private void UpdateTrainingLoss(double loss) => _trainingLoss += loss;
    private void UpdateValidationLoss(double loss) => _validationLoss += loss;
    
    private void UpdateWeights(List<double[,]> weightGradients, List<double[]> biasGradients)
    {
        var optimizerIndex = 0;
        // InputLayer -> HiddenLayer 
        for (var i = 0; i < InputLayer.Features.Count; i++)
        {
            var weights = InputLayer.Features[i].Weights;
            for (var j = 0; j < weights.Count; j++)
                weights[j] = _optimizers[optimizerIndex++].Update(weights[j], weightGradients[0][i,j]);
        }
        
        // Hidden layer -> Hidden layer.
        for (var i = 0; i < Layers.Count; i++)
        {
            for (var j = 0; j < Layers[i].Size(); j++)
            {
                var neuron = Layers[i].Neurons[j];
                var weights = neuron.Weights;
                
                var totalWeights = weights.Count;
                for (var w = 0; w < totalWeights; w++)
                      weights[w] = _optimizers[optimizerIndex++].Update(weights[w], weightGradients[i + 1][j, w]);
                
                neuron.Bias = _optimizers[optimizerIndex++].Update(neuron.Bias, biasGradients[i][j]);
            }
        }
    }
    
    private void SumGradients(List<Worker> workers, List<double[,]> weightGradients, List<double[]> biasGradients)
    {
        foreach (var worker in workers)
        {
            var inputLayerSize = InputLayer.Size();
            for (var i = 0; i <  inputLayerSize; i++)
                for (var j = 0; j < Layers[0].Size(); j++)
                    weightGradients[0][i,j] += worker.WeightGradients[0][i,j];
            
            for (var i = 0; i < Layers.Count - 1; i++)
                for (var j = 0; j < Layers[i].Size(); j++)
                    for (var k = 0; k < Layers[i + 1].Size(); k++)
                        weightGradients[i + 1][j, k] += worker.WeightGradients[i + 1][j, k];
                    
            for (var i = 0; i < worker.BiasGradients.Count; i++)
                for (var l = 0; l < worker.BiasGradients[i].Length; l++)
                    biasGradients[i][l] += worker.BiasGradients[i][l];
        }
    }
    
    private void AverageGradients(List<double[,]> weightGradients, List<double[]> biasGradients, int numberOfSamples)
    {
        var inputLayerSize = InputLayer.Size();
        for (var i = 0; i <  inputLayerSize; i++)
            for (var j = 0; j < Layers[0].Size(); j++)
                weightGradients[0][i,j] /= numberOfSamples;
            
        for (var i = 0; i < Layers.Count - 1; i++)
            for (var j = 0; j < Layers[i].Size(); j++)
                for (var k = 0; k < Layers[i + 1].Size(); k++)
                    weightGradients[i + 1][j, k] /= numberOfSamples;
        
        for (var i = 0; i < biasGradients.Count; i++)
            for (var l = 0; l < biasGradients[i].Length; l++)
                biasGradients[i][l] /= numberOfSamples;
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
    
    private void InitOptimizers()
    {
        var size = InputLayer.Size() * Layers[0].Size();

        for (var i = 0; i < Layers.Count - 1; i++)
            size += Layers[i].Size() *  Layers[i + 1].Size();

        size += Layers.Sum(i => i.Size());
        
        _optimizers = new IOptimizer[size];
        for (var i = 0; i < size; i++)
            _optimizers[i] = _optimizer.Clone();
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