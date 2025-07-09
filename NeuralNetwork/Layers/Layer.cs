using ML.NeuralNetwork.ActivationFunctions;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

public class Layer(int numberNeurons, Type activationFunction)
{
    public List<Neuron> Neurons { get; set; } = InitListWithNItems<Neuron>(numberNeurons);
    
    public ActivationFunctionBase ActivationFunction { get; set; } = Activator.CreateInstance(activationFunction) as ActivationFunctionBase 
                                                                     ?? throw new ArgumentException($"Activation function type has to be child class of {nameof(ActivationFunctionBase)}.)");
    public int Size() => Neurons.Count;
    
    public double MaxWeight() => Neurons.Max(n => n.Weights.Max());
    
    public double MaxBias() => Neurons.Max(n => n.Bias);
}

public class InputLayer(int numberFeatures)
{
    public List<Feature> Features { get; set; } = InitListWithNItems<Feature>(numberFeatures);
    
    public int Size() => Features.Count;
    
    public double MaxWeight() => Features.Max(n => n.Weights.Max());

}