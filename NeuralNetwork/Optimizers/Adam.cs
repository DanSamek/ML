namespace ML.NeuralNetwork.Optimizers;

/// <summary>
/// https://arxiv.org/pdf/1412.6980
/// </summary>
public class Adam : IOptimizer
{
    public class Config
    {
        public double Alpha { get; init; } = 0.001;
        public double Beta1 { get; init; } = 0.9;
        public double Beta2 { get; init; } = 0.999;
        public double Epsilon { get; init; } = 1e-8;
    }
    
    public required Config Configuration { get; init; }
    
    private double _momentum, _velocity;
    private double _beta1T = 1,  _beta2T = 1;
    
    public double Update(double parameter, double gradient)
    {
        _momentum = Beta1 * _momentum + (1 - Beta1) * gradient;
        _velocity = Beta2 * _velocity + (1 - Beta2) * double.Pow(gradient, 2);
        
        _beta1T = Beta1 * _beta1T;
        _beta2T = Beta2 * _beta2T;
        var corrMomentum = _momentum / (1 - _beta1T); 
        var corrVelocity = _velocity / (1 - _beta2T);
        
        var result = parameter - Alpha * corrMomentum / (double.Sqrt(corrVelocity) + Epsilon);
        return result;
    }

    public IOptimizer Clone()
    {
        var result = new Adam
        {
            Configuration = Configuration,
        };
        return result;
    }
    
    private double Beta1 => Configuration.Beta1;
    private double Beta2 => Configuration.Beta2;
    private double Alpha => Configuration.Alpha;
    private double Epsilon => Configuration.Epsilon;
}