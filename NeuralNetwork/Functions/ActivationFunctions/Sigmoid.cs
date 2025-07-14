namespace ML.NeuralNetwork.ActivationFunctions;

/// <summary>
/// https://en.wikipedia.org/wiki/Sigmoid_function
/// </summary>
public class Sigmoid : ActivationFunctionBase
{
    public override double Value(double x)  => 1 / (1 + Math.Exp(-x));
    
    public override double Derivative(double x)
    {
        var sigmoid = 1.0 / (1.0 + Math.Exp(-x));
        return sigmoid * (1.0 - sigmoid);
    }

    public override double RandomWeight(double inWeightCount, double outWeightCount)
    {
        var a = Math.Sqrt(6) / Math.Sqrt(inWeightCount + outWeightCount);
        return Random.Shared.NextDouble() * 2 * a - a;
    }
}