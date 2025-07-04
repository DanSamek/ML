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

    public override double RandomWeight(double inWeightCount, double outWeightCount)
    {
        var result = double.Sqrt(6) / double.Sqrt(inWeightCount + outWeightCount) * Random.Shared.NextDouble();
        var sign = Random.Shared.Next(0,1) == 0 ? -1 : 1;
        return result * sign;
    }
}