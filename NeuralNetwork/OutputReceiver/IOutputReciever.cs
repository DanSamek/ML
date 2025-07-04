namespace ML.NeuralNetwork.OutputReceiver;

/// <summary>
/// Neural network output interface.
/// </summary>
public interface IOutputReceiver
{
    /// <summary>
    /// Receives a current loss.
    /// </summary>
    public void Loss(double loss);
}