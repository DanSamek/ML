using System.Text;

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
    
    
    public static NeuralNetworkHelper.TrainingItem Parse(string line, int inputSize)
    {
        var splitLine = line.Split(" ");
        
        var input = new List<double>();
        for (var i = 0; i < inputSize; i++)
            input.Add(double.Parse(splitLine[i]));
        
        List<double> expected = [double.Parse(splitLine[^1])];
        var trainingItem = new NeuralNetworkHelper.TrainingItem(input, expected);
        return trainingItem;
    }
}