namespace ML.Perceptron;

public class Perceptron
{
    public class DatasetItem(List<double> features, int expected)
    {
        public List<double> Features { get; set; } = features;
        public int Expected { get; set; } = expected;
    }
    
    /// <summary>
    /// Training dataset.
    /// </summary>
    public List<DatasetItem> Dataset { get; init; }
    
    /// <summary>
    /// Weights of perceptron.
    /// </summary>
    public List<double> Weights { get; set; }
    
    public double Intercept { get; set;  }
    
    
    private bool _randomizeWeights;
    
    public Perceptron(List<DatasetItem> dataset, List<double> weights, double intercept)
    {
        Dataset = dataset;
        Weights = weights;
        Intercept = intercept;
        _randomizeWeights = false;
    }

    public Perceptron(List<DatasetItem> dataset)
    {
        Dataset = dataset;
        _randomizeWeights = true;
    }
    
    public void Train()
    {
        RandomizeWeights();
        while (RunEpoch()) { }
    }
    
    private bool RunEpoch()
    {
        var changedWeights = false;
        foreach (var item in Dataset)
        {
            // x^T * w + intercept.
            var vectorMult = item.Features.Select((item, idx) => item * Weights[idx]).Sum();
            vectorMult += Intercept;

            var result = Clip(vectorMult);

            if (result == item.Expected) continue;
            
            // If it was incorrect adjust weights.
            var addition = 2 * item.Expected - 1;
            var adjusted = item.Features.Select(i => i * addition).ToList();
            
            // w_i+1 = w_i + (2*Y - 1) * x
            for (var i = 0; i < Weights.Count; i++)
                Weights[i] += adjusted[i];
            
            // Intercept = Intercept + (2*Y - 1)
            Intercept += addition;
            changedWeights = true;
        }
        
        return changedWeights;
        
        static int Clip(double value) => value >= 0 ? 1 : 0;
    }
    
    private static double RandomDouble() => Random.Shared.NextDouble() * Random.Shared.Next(-100, 100); 
    private void RandomizeWeights()
    {
        if (!_randomizeWeights) return;
        
        Intercept = RandomDouble();
        for (var i = 0; i < Weights.Count; i++)
            Weights[i] = RandomDouble();
    }
}