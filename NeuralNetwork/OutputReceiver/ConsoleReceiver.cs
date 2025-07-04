namespace ML.NeuralNetwork.OutputReceiver;

public class ConsoleReceiver : IOutputReceiver
{
    public void Loss(double loss) => Console.WriteLine($"Current loss: {loss}");
    
    public void Gradients()
    {
        // TODO
    }
}