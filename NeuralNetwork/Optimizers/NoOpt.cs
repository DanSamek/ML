namespace ML.NeuralNetwork.Optimizers;

/// <summary>
/// Simple w = w - LR * gradient.
/// </summary>
public class NoOpt : IOptimizer
{
    public class Config
    {
        public double LearningRate { get; init; } = 0.1;   
    }
    
    public required Config Configuration { get; init; }
    
    public double Update(double gradient, double weight) => weight - Configuration.LearningRate * gradient;
}