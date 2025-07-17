using ML.NeuralNetwork.Configs;
using ML.NeuralNetwork.OutputReceiver;

class Program
{
    public static void Main(string[] args)
    {
        var nn = Chess.Create(4, "./test_dataset", "./nets/quant_net69.bin");
        
        nn.SetOutputReceiver(new ConsoleReceiver());
        var options = new TrainingOptions
        {
            NumberOfThreads = 1,
            NumEpochs = 20,
            BatchSize = 2048
        };
        nn.Train(options);
    }
}
