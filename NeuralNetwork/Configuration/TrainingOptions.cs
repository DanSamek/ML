namespace ML.NeuralNetwork;

public class TrainingOptions
{
    public double LearningRate { get; init; }
    public int NumEpochs { get; init; }
    public int BatchSize { get; init; }

    public int NumberOfThreads { get; set; }
    
    public TrainingOptions(int numEpochs, int batchSize,  double learningRate, int numberOfThreads)
    {
        NumEpochs = numEpochs;
        BatchSize = batchSize;
        LearningRate = learningRate;
        NumberOfThreads = numberOfThreads;
    }
}