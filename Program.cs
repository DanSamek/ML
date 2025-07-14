using ML.NeuralNetwork.Configs;
using ML.NeuralNetwork.OutputReceiver;

class Program
{
    public static void Main(string[] args)
    {
        var nn = Chess.Create(4, "./test_dataset");
        nn.SetOutputReceiver(new ConsoleReceiver());
        var options = new TrainingOptions
        {
            NumberOfThreads = 8,
            NumEpochs = 20,
            BatchSize = 128 
        };
        nn.Train(options);
    }
}
