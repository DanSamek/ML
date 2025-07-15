namespace ML.NeuralNetwork.OutputReceiver;

public class ConsoleReceiver : IOutputReceiver
{
    public void TrainingLoss(double loss)
    {
        
    }

    public void ValidationLoss(double loss) => Console.WriteLine($"Validation loss: {loss}");
    
    public void EpochCompleted(int epoch, double totalLoss) => Console.WriteLine($"Epoch: {epoch}, Total loss: {totalLoss}");
}