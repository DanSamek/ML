namespace ML.NeuralNetwork;

public static class NeuralNetworkHelper
{
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
    
    // q_val = round(value / scale), scale = max(|w|) / scale.
    public static int QuantizedValue(double value, double scale) => (int)double.Round(value / scale);
    public static double Scale(double max, double q) => max / q;
    
    
    public record TrainingItem(double[] Input, double[] Expected, bool Validation = false);
    public record ForwardResult(List<double> Output, List<double> Expected);

}