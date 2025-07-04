namespace ML.NeuralNetwork.ActivationFunctions;

/// <summary>
/// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
/// </summary>
public class RELU : ActivationFunctionBase
{
    public override double Value(double x) => double.Max(0, x);
    public override double Derivative(double x)  => x <= 0 ? 0 : 1;
    public override double RandomWeight(double inWeightCount, double outWeightCount) 
        => Random.Shared.NextDouble() * double.Sqrt(2 / inWeightCount);

}