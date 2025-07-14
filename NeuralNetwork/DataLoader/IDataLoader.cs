namespace ML.NeuralNetwork.Loader;

public interface IDataLoader
{
    /// <summary>
    /// Resets stream.
    /// </summary>
    public void Reset();

    /// <summary>
    /// Count lines in file.
    /// </summary>
    public long CountLines();
    
    /// <summary>
    /// Gets next item from a dataloader.
    /// </summary>
    /// <returns></returns>
    public NeuralNetworkHelper.TrainingItem? GetNext();

    /// <summary>
    /// Record for data loading from file.
    /// </summary>
    /// <param name="Line">Line from file.</param>
    /// <param name="Input">Array for input -- what will be forwarded in neural network.</param>
    /// <param name="Output">Array for output -- values for error calculations.</param>
    public record LoadContext(string Line, double[] Input, double[] Output);

}