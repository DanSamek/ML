using System.Text;
using NUnit.Framework;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork.Loader;

[TestFixture]
public class DataLoaderTests
{
    private readonly IReadOnlyList<(double, double, double)> _data = new List<(double, double, double)>()
    {
        (0.50,-0.506,0.4045),
        (1,1.9055,1.1404),
        (2.1540,1,2.5405),
        (5.45406054,3,5.405600440),
        (6,2.1206,1),
        (1.464647,3.450697,-5),
        (5.1,506.525056484,5.5)
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
    
    private TrainingItem Parse(string line)
    {
        var splitLine = line.Split(" ");
        List<double> input = [double.Parse(splitLine[0]), double.Parse(splitLine[1])]; 
        List<double> expected = [double.Parse(splitLine[^1])]; 
        var trainingItem = new TrainingItem(input, expected);
        return trainingItem;
    }
    
    [Test]
    public void ReadTest()
    {
        var fileName = CreateFile();
        var dataLoader = new DataLoader(fileName, Parse);
        
        var index = 0;
        while (true)
        {
            var item = dataLoader.GetNext();
            if (item is null)
                break;
            
            Assert.That(Math.Abs(_data[index].Item1 - item.Input[0]) < 1e-10);
            Assert.That(Math.Abs(_data[index].Item2 - item.Input[1]) < 1e-10);
            Assert.That(Math.Abs(_data[index].Item3 - item.Expected[0]) < 1e-10);
            index++;
        }
        
        File.Delete(fileName);
    }
}