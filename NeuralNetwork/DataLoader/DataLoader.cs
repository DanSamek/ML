using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork.Loader;

public class DataLoader
{
    /// <summary>
    /// Record for data loading from file.
    /// </summary>
    /// <param name="Line">Line from file.</param>
    /// <param name="Input">Array for input -- what will be forwarded in neural network.</param>
    /// <param name="Output">Array for output -- values for error calculations.</param>
    public record LoadContext(string Line, double[] Input, double[] Output);
    
    private readonly Action<LoadContext> _dataConverter;
    private readonly string _dataSource;
    private StreamReader? _streamReader;
    private readonly TrainingItem?[] _buffer;
    private readonly SortedSet<int> _notNullIndexes;
    private bool _forceSelected;

    private readonly int _inputLayerSize;
    private readonly int _outputLayerSize;
    /// <summary>
    /// .Ctor.
    /// </summary>
    /// <param name="dataSource">Path for dataset.</param>
    /// <param name="dataConverter">Custom function that will convert 1 line from dataset to "internal" structure.</param>
    /// <param name="inputLayerSize">Size of the input layer - (array size)</param>
    /// <param name="outputLayerSize">Size of the output layer - (array size)</param>
    /// <param name="maxLoadedItemsInMemory">How many items can be loaded in the memory - for randomization.</param>
    public DataLoader(string dataSource, 
                      Action<LoadContext> dataConverter,
                      int inputLayerSize, 
                      int outputLayerSize,
                      int maxLoadedItemsInMemory = 256)
    {
        _dataConverter = dataConverter;
        _dataSource = dataSource;
        _buffer = new TrainingItem[maxLoadedItemsInMemory];
        _notNullIndexes = [];
        _inputLayerSize = inputLayerSize;
        _outputLayerSize = outputLayerSize;
        Reset();
    }

    public void Reset()
    {
        ResetStream();
        Preload();
    }
    
    public long CountLines()
    {
        var totalLines = 0L;
        using var reader = new StreamReader(_dataSource);
        while (reader.ReadLine() != null)
            totalLines++;
        
        return totalLines;
    }
    
    public TrainingItem? GetNext()
    {
        if (_streamReader!.EndOfStream)
            return ForceSelect();
        
        var randomIndex = Random.Shared.Next(0, _buffer.Length);
        if (_buffer[randomIndex] == null)
            return ForceSelect(); // This branch will happen only if _streamReader.EndOfStream. 
        
        var entry = _buffer[randomIndex];
        _buffer[randomIndex] = GetNextEntry();
        return entry;
        
    }
    
    private void ResetStream()
    {
        _streamReader?.Close();
        _streamReader = new StreamReader(_dataSource);
    }
    
    private void Preload()
    {
        _notNullIndexes.Clear();
        _forceSelected = false;
        for (var i = 0; i < _buffer.Length; i++)
            _buffer[i] = GetNextEntry();
    }

    private TrainingItem? GetNextEntry()
    {
        if (_streamReader!.EndOfStream)
            return null;
        
        var inputLayer = NeuralNetworkArrayPool.Rent(_inputLayerSize, true);
        var outputLayer = NeuralNetworkArrayPool.Rent(_outputLayerSize, false);
        _dataConverter(new LoadContext(_streamReader.ReadLine()!, inputLayer, outputLayer));

        var result = new TrainingItem(inputLayer, outputLayer);
        return result;
    }
    
    private TrainingItem? ForceSelect()
    {
        if (!_forceSelected)
        {
            for (var i = 0; i < _buffer.Length; i++)
                if (_buffer[i] != null)
                    _notNullIndexes.Add(i);
        }
            
        if (_notNullIndexes.Count == 0)
            return null;
            
        var randomIndex = Random.Shared.Next(0, _notNullIndexes.Count);
        var entryIndex = _notNullIndexes.ElementAt(randomIndex);
        var entry = _buffer[entryIndex];
            
        _notNullIndexes.Remove(entryIndex);
        _buffer[entryIndex] = null;
        _forceSelected = true;
        return entry;
    }
}