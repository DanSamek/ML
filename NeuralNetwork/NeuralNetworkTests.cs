using System.Text;
using ML.NeuralNetwork.Loader;
using NUnit.Framework;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

[TestFixture]
public class NeuralNetworkTests
{
    private readonly IReadOnlyList<(double, double, double)> _data = new List<(double, double, double)>()
    {
        (0.5,-0.5,0.4),
        (1,1.9,1.14),
        (2.1,1,2.5405),
        (5.4,3,5.4),
        (6,2.12,1),
        (1.46,3.45,-5),
        (5.1,5.2,5.5)
    };
    
    private string CreateFile()
    {
        var fileName = Guid.NewGuid().ToString();
        var stringBuilder = new StringBuilder();
        foreach (var item in _data)
        {
            var line = $"{item.Item1} {item.Item2} {item.Item3}";
            stringBuilder.AppendLine(line);
        }

        var toSave = stringBuilder.ToString();
        File.WriteAllText(fileName, toSave);
        return fileName;
    }
    
    private static TrainingItem Parse(string line)
    {
        var splitLine = line.Split(" ");
        List<double> input = [double.Parse(splitLine[0]), double.Parse(splitLine[1])]; 
        List<double> expected = [double.Parse(splitLine[^1])]; 
        var trainingItem = new TrainingItem(input, expected);
        return trainingItem;
    }

    // Absolute error.
    private double LossFunction(ForwardResult result) => Math.Abs(result.Expected[0] - result.Output[0]);
    
    private const string NN_NAME = "test.bin";
    
    [Test]
    public void LoadTest()
    {
        var fileName = CreateFile();
        var nn = new NeuralNetwork()
            .AddInputLayer(2)
            .AddHiddenLayer(1, value => value)
            .SetLossFunction(LossFunction)
            .SetDataLoader(new DataLoader(fileName, Parse)) 
            .Build();
        
        nn.InitializeRandom();
        nn.Save(NN_NAME);
        nn.Load(NN_NAME);
        
        // TODO somehow compare weights.
        // {get; private set} maybe.
        
        File.Delete(fileName);
        File.Delete(NN_NAME);
    }
    
    /// <summary>
    /// TODO!
    /// First needed load net from file -- custom weights.
    /// </summary>
    [Test]
    public void ForwardTest()
    {
        var fileName = CreateFile();
        var nn = new NeuralNetwork()
                    .AddInputLayer(2)
                    .AddHiddenLayer(1, value => value)
                    .SetLossFunction(LossFunction)
                    .SetDataLoader(new DataLoader(fileName, Parse)) 
                    .Build();

        var options = new TrainingOptions();
        nn.Train(options);
        File.Delete(fileName);
    }
}