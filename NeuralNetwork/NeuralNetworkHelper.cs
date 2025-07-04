namespace ML.NeuralNetwork;

public static class NeuralNetworkHelper
{
    /// <summary>
    /// Used definition:
    /// lim h->0 f(x + h) - f(x) / h.
    /// </summary>
    public static double Derivative(Func<double, double> function, double x)
    {
        const double h = 0.000001;
        var derivative = (function(x + h) - function(x)) / h;
        return derivative;
    }

    public static List<T> InitListWithNItems<T>(int n) where T : new()
    {
        var result = new List<T>();
        for (var i = 0; i < n; i++)
            result.Add(new T());
        
        return result;
    }
    
    public static (List<double[,]>, List<double[]>) CreateArraysForGradients(NeuralNetwork neuralNetwork)
    {
        var inputLayerSize = neuralNetwork.InputLayer.Size();
        var weightGradients = new List<double[,]>();
        
        weightGradients.Add(new double[inputLayerSize, neuralNetwork.Layers[0].Size()]);
        for (var i = 0; i < neuralNetwork.Layers.Count - 1; i++)
            weightGradients.Add(new double[neuralNetwork.Layers[i].Size(), neuralNetwork.Layers[i + 1].Size()]);
        
        var biasGradients = neuralNetwork.Layers
            .Select(l => new double[l.Size()])
            .ToList();
        
        return (weightGradients, biasGradients);
    }
    
    public record TrainingItem(List<double> Input, List<double> Expected, bool Validation = false);
    public record ForwardResult(List<double> Output, List<double> Expected);

}