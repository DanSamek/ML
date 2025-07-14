using System.Text;
using ML.NeuralNetwork.Loader;

namespace ML.NeuralNetwork;

public static class NeuralNetworkTestBase
{
    public static string CreateFile(IReadOnlyList<List<double>> data)
    {
        var fileName = Guid.NewGuid().ToString();
        var stringBuilder = new StringBuilder();
        var lineSb = new StringBuilder();
        
        foreach (var item in data)
        {
            lineSb.Clear();
            for (var i = 0; i < item.Count - 1; i++)
                lineSb.Append($"{item[i]} ");
            lineSb.Append(item[^1]);
            
            var line = lineSb.ToString();
            stringBuilder.AppendLine(line);
        }

        var toSave = stringBuilder.ToString();
        File.WriteAllText(fileName, toSave);
        return fileName;
    }
    
    
    public static void Parse(IDataLoader.LoadContext context, int inputSize)
    {
        var splitLine = context.Line.Split(" ");
        for (var i = 0; i < inputSize; i++) 
            context.Input[i] = double.Parse(splitLine[i]);

        context.Output[0] = double.Parse(splitLine[^1]);
    }
}