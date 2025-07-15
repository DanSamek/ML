namespace ML.NeuralNetwork;

public class TrainingOptions
{
    /// <summary>
    ///  Number of epochs.
    /// </summary>
    public int NumEpochs { get; init; } = 1;
    
    /// <summary>
    /// Batch size.
    /// </summary>
    public int BatchSize { get; init; } = 1;
    
    /// <summary>
    /// Number of threads, that will be used for training.
    /// </summary>
    public int NumberOfThreads { get; init; } = 1;
    
    /// <summary>
    /// After how many epochs will be neural network saved.
    /// </summary>
    public int SaveRate { get; init; } = 1;
    
    /// <summary>
    /// "Template" for neural network saves.
    /// Input is epoch number.
    /// </summary>
    public Func<int, string> NnNameGetter { get; init; } = epoch => $"nn_{epoch}.bin";
}