namespace ML.NeuralNetwork;

public class Neuron : Feature
{
    public double Bias { get; set; }
    
    public double Sum { get; set; }
    
    // Value after activation function.
    public double ActivatedValue { get; set; }
}