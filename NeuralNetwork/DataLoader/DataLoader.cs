using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork.Loader;

public class DataLoader
{
    private readonly Func<string, TrainingItem> _dataConverter;
    private readonly string _dataSource;
    private StreamReader _streamReader;
    
    public DataLoader(string dataSource,  Func<string, TrainingItem> dataConverter)
    {
        _dataConverter = dataConverter;
        _dataSource = dataSource;
        _streamReader = new StreamReader(_dataSource);
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
        return _streamReader.EndOfStream ? null : _dataConverter(_streamReader.ReadLine()!);
    }
    
    public void ResetStream()
    {
        _streamReader.Close();
        _streamReader = new StreamReader(_dataSource);
    }
}