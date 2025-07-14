namespace ML.NeuralNetwork.ActivationFunctions;

public class Tanh : ActivationFunctionBase
{
    public override double Derivative(double x)  => 1 - Math.Pow(Math.Tanh(x), 2);
    public override double Value(double x) => Math.Tanh(x);

    public override double RandomWeight(double inWeightCount, double outWeightCount)
    {
        var a = Math.Sqrt(6) / Math.Sqrt(inWeightCount + outWeightCount);
        return Random.Shared.NextDouble() * 2 * a - a;
    }
}