namespace ML.Perceptron;

public class DatasetItem<T>(List<T> features, int expected)
{
    public List<T> Features { get; set; } = features;
    public int Expected { get; set; } = expected;
}

public class Perceptron<T>
{
    /// <summary>
    /// Training dataset.
    /// </summary>
    public List<DatasetItem<T>> Dataset { get; init; }
    
    /// <summary>
    /// Weights of perceptron.
    /// </summary>
    public List<double> Weights { get; set; }
    
    /// <summary>
    /// "Bias"
    /// </summary>
    public double Intercept { get; set;  }
    
    private readonly int _maxEpoch;
    private readonly Func<T, double> _doubleConverter;
    
    public Perceptron(List<DatasetItem<T>> dataset,
                      List<double> weights,
                      double intercept,
                      Func<T, double> doubleConverter,
                      int maxEpoch = 5000)
    {
        Dataset = dataset;
        Weights = weights;
        Intercept = intercept;
        _maxEpoch = maxEpoch;
        _doubleConverter =  doubleConverter;
    }

    public Perceptron(List<DatasetItem<T>> dataset, 
                      Func<T, double> doubleConverter,
                      int maxEpoch = 5000)
    {
        Dataset = dataset;
        Weights = new List<double>( new double[dataset[0].Features.Count]);
        _doubleConverter = doubleConverter;
        _maxEpoch = maxEpoch;
        Intercept = 0;
    }
    
    public void Train()
    {
        for (var epoch = 0; epoch < _maxEpoch; epoch++)
            if (RunEpoch())
                break;
    }
    
    /*
     * Algorithm is simple:
     *      Forward input vector -> Clip value.
     *      If clipped value != expected, we adjust weights:
     *          w = w + (2 * expected - 1) * x.
     *          intercept = intercept + (2 * expected - 1)
     */
    private bool RunEpoch()
    {
        var numberOfErrors = 0;
        
        foreach (var item in Dataset)
        {
            // x^T * w + intercept.
            var vectorMult = 0.0;
            for (var i = 0; i < item.Features.Count; i++)
                vectorMult += _doubleConverter.Invoke(item.Features[i]) * Weights[i];
            
            vectorMult += Intercept;

            var result = Clip(vectorMult);

            if (result == item.Expected) continue;
            
            // If it was incorrect adjust weights.
            var addition = 2 * item.Expected - 1;
            
            // w_i+1 = w_i + (2*Y - 1) * x
            for (var i = 0; i < Weights.Count; i++)
                Weights[i] += _doubleConverter.Invoke(item.Features[i]) * addition;
            
            // Intercept = Intercept + (2*Y - 1)
            Intercept += addition;
            numberOfErrors++;
        }
        
        Console.WriteLine($"loss: {numberOfErrors * 1.0 / Dataset.Count}");
        
        return numberOfErrors == 0;
        
        static int Clip(double value) => value >= 0 ? 1 : 0;
    }
}