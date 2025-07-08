namespace ML.NeuralNetwork;

public class TrainingOptions
{
    public int NumEpochs { get; init; } = 1;
    public int BatchSize { get; init; } = 1;

    public int NumberOfThreads { get; set; } = 1;
    
    public TrainingOptions(int numEpochs, int batchSize, int numberOfThreads)
    {
        NumEpochs = numEpochs;
        BatchSize = batchSize;
        NumberOfThreads = numberOfThreads;
    }

    public TrainingOptions(){}
}