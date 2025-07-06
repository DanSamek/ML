namespace ML.NeuralNetwork.Optimizers;

/// <summary>
/// Base class for all optimizers.
/// </summary>
public interface IOptimizer
{
    /// <summary>
    /// Updates weight with some optimization technique.
    /// </summary>
    /// <param name="gradient">Gradient for weight.</param>
    /// <param name="weight">Weight.</param>
    /// <returns></returns>
    public double Update(double gradient, double weight);
}