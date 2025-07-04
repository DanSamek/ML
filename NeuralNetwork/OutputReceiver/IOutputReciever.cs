namespace ML.NeuralNetwork.OutputReceiver;

/// <summary>
/// Neural network output interface.
/// </summary>
public interface IOutputReceiver
{
    /// <summary>
    /// Receives a current loss.
    /// </summary>
    public void TrainingLoss(double loss);
    
    /// <summary>
    /// Receives a validation loss.
    /// </summary>
    public void ValidationLoss(double loss);
}