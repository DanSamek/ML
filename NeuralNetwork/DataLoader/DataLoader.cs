using static ML.NeuralNetwork.Loader.IDataLoader;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork.Loader;

/// <summary>
/// Dataloader implementation without shuffling. 
/// </summary>
public class DataLoader : IDataLoader
{
    private readonly string _dataSource;
    protected StreamReader? StreamReader;
    private readonly Action<LoadContext> _dataConverter;
    private readonly int _inputLayerSize;
    private readonly int _outputLayerSize;
    
    /// <summary>
    /// .Ctor.
    /// </summary>
    /// <param name="dataSource">Path for dataset.</param>
    /// <param name="dataConverter">Custom function that will convert 1 line from dataset to "internal" structure.</param>
    /// <param name="inputLayerSize">Size of the input layer - (array size)</param>
    /// <param name="outputLayerSize">Size of the output layer - (array size)</param>
    public DataLoader(string dataSource, Action<LoadContext> dataConverter, int inputLayerSize, int outputLayerSize)
    {
        _dataSource = dataSource;
        _dataConverter = dataConverter;
        _inputLayerSize = inputLayerSize;
        _outputLayerSize = outputLayerSize;
        StreamReader = new StreamReader(_dataSource);
    }
    
    /// <inheritdoc cref="IDataLoader.Reset"/>
    public virtual void Reset()
    {
        StreamReader?.Close();
        StreamReader = new StreamReader(_dataSource);
    }
    
    /// <inheritdoc cref="IDataLoader.CountLines"/> 
    public long CountLines()
    {
        var totalLines = 0L;
        using var reader = new StreamReader(_dataSource);
        while (reader.ReadLine() != null)
            totalLines++;
        
        return totalLines;
    }

    /// <summary>
    /// Returns a next item from a buffer.
    /// </summary>
    public virtual TrainingItem? GetNext() => GetNextEntry();
    
    /// <summary>
    /// Gets next entry from a buffer.
    /// </summary>
    protected TrainingItem? GetNextEntry()
    {
        if (StreamReader!.EndOfStream)
            return null;
        
        var inputLayer = NeuralNetworkArrayPool.Rent(_inputLayerSize, true);
        var outputLayer = NeuralNetworkArrayPool.Rent(_outputLayerSize, false);
        _dataConverter(new LoadContext(StreamReader.ReadLine()!, inputLayer, outputLayer));

        var result = new TrainingItem(inputLayer, outputLayer);
        return result;
    }

}