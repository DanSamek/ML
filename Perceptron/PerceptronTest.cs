namespace ML.Perceptron;

public static class PerceptronTest
{
    private const string PATH = "./Perceptron/positions_shuffled.txt";

    private static readonly Dictionary<char, int> PIECE_INT = new() 
    {
        {'p', 0},
        {'n', 1},
        {'b', 2},
        {'r', 3},
        {'q', 4},
        {'k', 5}
    };
    
    private static void ParseFen(bool[] features, string fen)
    {
        var rows = fen.Split("/");
        for (var row = 0; row < rows.Length; row++)
        {
            var idx = 0;
            foreach (var item in rows[row])
            {
                if (int.TryParse(item.ToString(), out var value))
                {
                    idx += value;
                    continue;
                }
                
                var isWhite = char.IsUpper(item);
                var pieceValue = PIECE_INT[char.ToLower(item)] + (isWhite ? 0 : 6);
                features[(row * 8 + idx) * pieceValue] = true;

                idx++;
            }
        }
    }
    
    private static List<DatasetItem<bool>> LoadDataset()
    {
        var result = new List<DatasetItem<bool>>();
        
        using var stream = new StreamReader(PATH);
        while (!stream.EndOfStream)
        {
            if (result.Count >= 2000)
                break;
            
            var line = stream.ReadLine()!;
            
            var features = new bool[768];
            var splitLine = line.Split("|");
            var fen =  splitLine[0];
            ParseFen(features, fen);
            var whiteWinning = int.Parse(splitLine[1]) > 0 ? 1 : 0;
            result.Add(new DatasetItem<bool>(features.ToList(), whiteWinning));
        }
        
        return result;
    }
    
    /// <summary>
    /// Test if perceptron can determine fully winning position for white in chess
    ///     - The problem is "pretty hard" for only perceptron, only test, how good/bad it is before neural networks :).
    /// 768 (12*64) -> 1
    /// </summary>
    public static void PositionTest()
    {
        var dataset = LoadDataset();
        var perceptron = new Perceptron<bool>(dataset, item => item ? 1.0 : 0.0, 5000);
        perceptron.Train();

        using var fs = new FileStream("chess_perceptron.bin", FileMode.Create);
        var bytes = BitConverter.GetBytes(perceptron.Intercept);
        fs.Write(bytes);
        fs.Write("\0\0\0"u8);
        foreach (var weight in perceptron.Weights)
        {
            bytes = BitConverter.GetBytes(weight);
            fs.Write(bytes);
            fs.Write("\0\0\0"u8);
        }
    }
}