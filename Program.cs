
using ML.NeuralNetwork;
using ML.NeuralNetwork.Loader;

class Program
{
    public static void Main(string[] args)
    {
        var nn = new NeuralNetwork().AddInputLayer(20)
                                    .AddHiddenLayer(100, ActivationFunctions.RELU)
                                    .SetLossFunction((result) => Math.Pow(result.Output[0] - result.Expected[0], 2))
                                    .Build();
        
        nn.Train(null);
        nn.UseQuantization();
        nn.Save("net.bin");
    }
}
