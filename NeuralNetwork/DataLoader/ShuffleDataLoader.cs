using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork.Loader;

/// <summary>
/// Dataloader implementation that selects randomly from a buffer. 
/// </summary>
public class ShuffleDataLoader : DataLoader
{
    private readonly TrainingItem?[] _buffer;
    private readonly SortedSet<int> _notNullIndexes;
    private bool _forceSelected;

    /// <summary>
    /// .Ctor.
    /// </summary>
    /// <param name="dataSource">Path for dataset.</param>
    /// <param name="dataConverter">Custom function that will convert 1 line from dataset to "internal" structure.</param>
    /// <param name="inputLayerSize">Size of the input layer - (array size)</param>
    /// <param name="outputLayerSize">Size of the output layer - (array size)</param>
    /// <param name="maxLoadedItemsInMemory">How many items can be loaded in the memory - for randomization.</param>
    public ShuffleDataLoader(string dataSource,
        Action<IDataLoader.LoadContext> dataConverter,
        int inputLayerSize,
        int outputLayerSize,
        int maxLoadedItemsInMemory = 256) : base(dataSource, dataConverter, inputLayerSize, outputLayerSize)
    {
        _buffer = new TrainingItem[maxLoadedItemsInMemory];
        _notNullIndexes = [];
        Preload();
    }
    
    public override void Reset()
    {
        base.Reset();
        Preload();
    }
    
    public override TrainingItem? GetNext()
    {
        if (StreamReader!.EndOfStream)
            return ForceSelect();
        
        var randomIndex = Random.Shared.Next(0, _buffer.Length);
        if (_buffer[randomIndex] == null)
            return ForceSelect(); // This branch will happen only if _streamReader.EndOfStream. 
        
        var entry = _buffer[randomIndex];
        _buffer[randomIndex] = GetNextEntry();
        return entry;
        
    }
    
    private void Preload()
    {
        _notNullIndexes.Clear();
        _forceSelected = false;
        for (var i = 0; i < _buffer.Length; i++)
            _buffer[i] = GetNextEntry();
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