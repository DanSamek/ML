namespace ML.NeuralNetwork.LossFunctions;

public class MSE : LossFunctionBase
{
    public double Value(double current, double expected) => Math.Pow(expected - current, 2);
    
    public double Derivative(double current, double expected) => 2 * (current - expected);
}