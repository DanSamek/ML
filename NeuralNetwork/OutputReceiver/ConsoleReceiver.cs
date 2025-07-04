namespace ML.NeuralNetwork.OutputReceiver;

public class ConsoleReceiver : IOutputReceiver
{
    public void TrainingLoss(double loss) => Console.WriteLine($"Current loss: {loss}");
    
    public void ValidationLoss(double loss) => Console.WriteLine($"Validation loss: {loss}");
    
    public void Gradients()
    {
        // TODO
    }
}