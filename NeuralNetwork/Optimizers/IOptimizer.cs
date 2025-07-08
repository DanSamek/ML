namespace ML.NeuralNetwork.Optimizers;

/// <summary>
/// Base class for all optimizers.
/// </summary>
public interface IOptimizer
{
    /// <summary>
    /// Updates param with some optimization technique.
    /// </summary>
    /// <param name="parameter">Parameter (Weight, bias).</param>
    /// <param name="gradient">Gradient for the parameter.</param>
    /// <returns></returns>
    public double Update(double parameter, double gradient);
    
    /// <summary>
    /// Clones an current instance.  
    /// </summary>
    /// <returns></returns>
    public IOptimizer Clone();
}