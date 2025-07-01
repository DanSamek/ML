namespace ML.NeuralNetwork.Loader;

public class DataLoader
{
    private readonly Func<string, (List<double> input, List<double> expected)> _dataConverter;
    private readonly string _dataSource;
    public DataLoader(string dataSource,  Func<string, (List<double> input, List<double> expected)> dataConverter)
    {
        _dataConverter = dataConverter;
        _dataSource = dataSource;
    }
    
    public long CountLines()
    {
        var totalLines = 0L;
        using var reader = new StreamReader(_dataSource);
        while (reader.ReadLine() != null)
            totalLines++;
        
        return totalLines;
    }
    
    public IEnumerable<(List<double> input, List<double> expected)> GetEnumerator()
    {
        using var reader = new StreamReader(_dataSource);
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            yield return _dataConverter(line!);
        }
    }
}