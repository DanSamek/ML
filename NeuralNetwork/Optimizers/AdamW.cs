namespace ML.NeuralNetwork.Optimizers;

public class AdamW : IOptimizer
{
    public class Config : Adam.Config
    {
        public double WeightDecay { get; init; } = 0.01; 
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
        
        var adamUpdate =  Alpha * corrMomentum / (double.Sqrt(corrVelocity) + Epsilon);
        var decoupledWeightDecay = Alpha * WeightDecay * parameter;
        var result = parameter - adamUpdate - decoupledWeightDecay;
        return result;
    }

    public IOptimizer Clone()
    {
        var result = new AdamW
        {
            Configuration = Configuration,
        };
        return result;
    }
    
    private double Beta1 => Configuration.Beta1;
    private double Beta2 => Configuration.Beta2;
    private double Alpha => Configuration.Alpha;
    private double Epsilon => Configuration.Epsilon;
    private double WeightDecay => Configuration.WeightDecay;
}