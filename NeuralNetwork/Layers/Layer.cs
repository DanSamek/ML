using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

public class Layer(int numberNeurons, Func<double, double> activationFunction)
{
    public List<Neuron> Neurons { get; set; } = InitListWithNItems<Neuron>(numberNeurons);
    
    public Func<double, double> ActivationFunction { get; set; } =  activationFunction;
    
    public int Size() => Neurons.Count;
    
    /// <summary>
    /// Runs activation function for all neurons.
    /// </summary>
    public void Activate() => Neurons.ForEach(n => n.Sum = ActivationFunction(n.Sum));
}

public class InputLayer(int numberFeatures)
{
    public List<Feature> Features { get; set; } = InitListWithNItems<Feature>(numberFeatures);
    
    public int Size() => Features.Count;
}