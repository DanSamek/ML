using ML.NeuralNetwork.Loader;

namespace ML.NeuralNetwork;

public class TrainingOptions<T>
{
    public double LearningRate { get; init; }
    public int NumEpochs { get; init; }
    public int NumBatches { get; init; }
    
    public DataLoader<T> DataLoader { get; init; }

    public TrainingOptions(DataLoader<T> dataLoader, int numEpochs, int numBatches,  double learningRate)
    {
        DataLoader = dataLoader;
        NumEpochs = numEpochs;
        NumBatches = numBatches;
        LearningRate = learningRate;
    }
}