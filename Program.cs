
using ML.NeuralNetwork;

class Program
{
    public static void Main(string[] args)
    {
        var nn = new NeuralNetwork().AddInputLayer(20)
                                    .AddHiddenLayer(100, ActivationFunctions.RELU)
                                    .SetLossFunction((outputLayer, expected) => Math.Pow(outputLayer - expected, 2))
                                    .Build();
        
        nn.Train<double>(null);
        nn.UseQuantization();
        nn.Save("net.bin");
    }
}
