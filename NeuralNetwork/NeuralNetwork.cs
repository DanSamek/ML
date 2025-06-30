using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

public class NeuralNetwork
{
    private List<Layer> _layers;
    private InputLayer _inputLayer;
    private Func<double, double, double> _lossFunction;
    
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
    public NeuralNetwork SetLossFunction(Func<double, double, double> lossFunction)
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
    public void Train<T>(TrainingOptions<T> trainingOptions)
    {
        var biggestLayerSize = Math.Max(_inputLayer.Size(), _layers.Max(l => l.Size()));
        
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

    private void ResetNeuronSums() => _layers.ForEach(l => l.Neurons.ForEach(n => n.Sum = 0));
}