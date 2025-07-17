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
    /// Input is epoch number and if its for quantization.
    /// It saves both non quantized is used for loading.
    /// </summary>
    public Func<int, bool, string> NnNameGetter { get; init; } = (epoch, quant) => quant ? $"quantized_nn_{epoch}.bin" : $"nn_{epoch}.bin";


}