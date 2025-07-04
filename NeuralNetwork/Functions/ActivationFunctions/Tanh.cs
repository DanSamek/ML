namespace ML.NeuralNetwork.ActivationFunctions;

public class Tanh : ActivationFunctionBase
{
    public override double Derivative(double x)  => 1 - Math.Pow(Math.Tanh(x), 2);
    public override double Value(double x) => Math.Tanh(x);

    public override double RandomWeight(double inWeightCount, double outWeightCount)
    {
        var result =  1 / double.Sqrt(inWeightCount) * Random.Shared.NextDouble();
        var sign = Random.Shared.Next(0,1) == 0 ? -1 : 1;
        return result * sign;
    }
}