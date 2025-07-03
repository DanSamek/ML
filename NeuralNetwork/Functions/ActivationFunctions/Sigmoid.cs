namespace ML.NeuralNetwork.ActivationFunctions;

/// <summary>
/// https://en.wikipedia.org/wiki/Sigmoid_function
/// </summary>
public class Sigmoid : ActivationFunctionBase
{
    public override double Value(double x)  => 1 / (1 + Math.Exp(-x));
    
    public override double Derivative(double x)
    {
        var exp = Math.Exp(-x);
        return exp / Math.Pow(exp + 1, 2);
    }
}