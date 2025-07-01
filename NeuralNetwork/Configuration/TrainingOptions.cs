namespace ML.NeuralNetwork;

public class TrainingOptions
{
    public double LearningRate { get; init; } = 0.001;
    public int NumEpochs { get; init; } = 1;
    public int BatchSize { get; init; } = 1;

    public int NumberOfThreads { get; set; } = 1;
    
    public TrainingOptions(int numEpochs, int batchSize,  double learningRate, int numberOfThreads)
    {
        NumEpochs = numEpochs;
        BatchSize = batchSize;
        LearningRate = learningRate;
        NumberOfThreads = numberOfThreads;
    }
}