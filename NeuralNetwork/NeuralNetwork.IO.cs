using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

public partial class NeuralNetwork
{
    private void SaveQuantized(string path)
    {
        if (_scales.Count == 0) return;
        
        using var binaryWriter = new BinaryWriter(File.Open(path, FileMode.Create));
        Span<double> calculatedScales = stackalloc double[_scales.Count];
        CalculateScales(calculatedScales);

        foreach (var weight in InputLayer.Features.SelectMany(feature => feature.Weights))
        {
            var value = QuantizedValue(weight, calculatedScales[0], _scales[0]);
            binaryWriter.Write(value);
        }
        
        for (var i = 0; i < Layers.Count; i++)
        {
            foreach (var neuron in Layers[i].Neurons)
            {
                var value = QuantizedValue(neuron.Bias, calculatedScales[i], _scales[i]);
                binaryWriter.Write(value);
                foreach (var weight in neuron.Weights)
                {
                    value = QuantizedValue(weight, calculatedScales[i + 1], _scales[i + 1]);
                    binaryWriter.Write(value);
                }
            }
        }
    }

    private void SaveNormal(string path)
    {
        using var binaryStream = new BinaryWriter(File.Open(path, FileMode.Create));
        foreach (var weight in InputLayer.Features.SelectMany(feature => feature.Weights))
            binaryStream.Write(weight);
            
        foreach (var neuron in Layers.SelectMany(l => l.Neurons))
        {
            binaryStream.Write(neuron.Bias);
            foreach (var weight in neuron.Weights)
                binaryStream.Write(weight);
        }
        binaryStream.Close();
    }

    private void LoadNormal(string path, bool quantized = false)
    {
        using var binaryStream = new BinaryReader(File.Open(path, FileMode.Open));
        foreach (var feature in InputLayer.Features)
        {
            for (var i = 0; i < feature.Weights.Count; i++)
                feature.Weights[i] = quantized ? binaryStream.ReadInt32() : binaryStream.ReadDouble();
        }
        
        foreach (var neuron in Layers.SelectMany(l => l.Neurons))
        {
            neuron.Bias =  quantized ? binaryStream.ReadInt32() : binaryStream.ReadDouble();
            for (var i = 0; i < neuron.Weights.Count; i++)
                neuron.Weights[i] = quantized ? binaryStream.ReadInt32() : binaryStream.ReadDouble();
        }
        
        binaryStream.Close();
    }

    private void CalculateScales(Span<double> calculatedScales)
    {
        var layerMaxWeight = double.Max(InputLayer.MaxWeight(), Layers[0].MaxBias());
        calculatedScales[0] = Scale(double.Abs(layerMaxWeight), _scales[0]);
        for (var i = 0; i < Layers.Count - 1; i++)
        {
            layerMaxWeight = double.Max(Layers[i].MaxWeight(), Layers[i + 1].MaxBias());
            calculatedScales[i + 1] = Scale(double.Abs(layerMaxWeight), _scales[i + 1]);
        }
    }
}