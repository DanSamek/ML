namespace ML.NeuralNetwork.LossFunctions;

public interface LossFunctionBase
{
    /// <summary>
    /// Activation function value. 
    /// </summary>
    /// <param name="current">Network output value</param>
    /// <param name="expected">Value from dataset</param>
    /// <returns></returns>
    public double Value(double current, double expected);
    
    /// <summary>
    /// Derivation of the activation function.
    /// (Partial derivative with respect to current value).
    /// </summary>
    public double Derivative(double current, double expected);
}