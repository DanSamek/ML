namespace ML.NeuralNetwork.Optimizers;

/// <summary>
/// Simple w = w - LR * gradient.
/// </summary>
public class Simple : IOptimizer
{
    public class Config
    {
        public double LearningRate { get; init; } = 0.1;   
    }
    
    public required Config Configuration { get; init; }
    
    public double Update(double parameter, double gradient) => parameter - Configuration.LearningRate * gradient;
    
    public IOptimizer Clone()
    {
        var result = new Simple
        {
            Configuration = Configuration
        };
        return result;
    }
}