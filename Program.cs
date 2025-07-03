
using System.Text;
using ML.NeuralNetwork;
using ML.NeuralNetwork.ActivationFunctions;
using ML.NeuralNetwork.Loader;
using ML.NeuralNetwork.LossFunctions;
using static ML.NeuralNetwork.NeuralNetworkHelper;

class Program
{
    private static readonly IReadOnlyList<(double, double, double)> _data = new List<(double, double, double)>()
    {
        (0.5,-0.5,0.4),
        (1,1.9,1.14),
        (2.1,1,2.5405),
        (5.4,3,5.4),
        (6,2.12,1),
        (1.46,3.45,-5),
        (5.1,5.2,5.5)
    };
    
    private static string CreateFile()
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

    private class SimpleActivationFunction : ActivationFunctionBase
    {
        public override double Value(double x) => x;
    }
    
    public static void Main(string[] args)
    {
        for (var i = 0; i < 1000; i++){
            var fileName = CreateFile();
            var nn = new NeuralNetwork()
                .AddInputLayer(2)
                .AddLayer(8, typeof(SimpleActivationFunction))
                .AddLayer(4, typeof(SimpleActivationFunction))
                .AddLayer(1, typeof(SimpleActivationFunction))
                .SetLossFunction(typeof(MAE))
                .SetDataLoader(new DataLoader(fileName, Parse)) 
                .Build();
        
            var options = new TrainingOptions
            {
                NumberOfThreads = 16
            };
            nn.Train(options);
            File.Delete(fileName);
        }
    }
}
