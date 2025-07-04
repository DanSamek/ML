namespace ML.NeuralNetwork.ActivationFunctions;

public abstract class ActivationFunctionBase
{
    /// <summary>
    /// Function value.
    /// </summary>
    public abstract double Value(double x);
    
    private const double H = 0.00001;
    /// <summary>
    /// Derivation of the function.
    /// </summary>
    public virtual double Derivative(double x)
    {
        // lim x->0 (f(x+h) - f(x)) / h
        var derivative = (Value(x + H) - Value(x)) / H;
        return derivative;
    }
    
    /// <summary>
    /// Generates random weight.
    /// </summary>
    /// <param name="inWeightCount">Number of inputs</param>
    /// <param name="outWeightCount">Number of outputs.</param>
    /// <returns></returns>
    public abstract double RandomWeight(double inWeightCount, double outWeightCount);
}