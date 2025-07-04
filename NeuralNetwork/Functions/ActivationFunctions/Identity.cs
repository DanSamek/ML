namespace ML.NeuralNetwork.ActivationFunctions;

public class Identity : ActivationFunctionBase
{
    public override double Value(double x) => x;
    public override double Derivative(double x) => 1;
    public override double RandomWeight(double inWeightCount, double outWeightCount) => Random.Shared.NextDouble();
}