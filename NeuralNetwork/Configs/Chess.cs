using ML.NeuralNetwork.ActivationFunctions;
using ML.NeuralNetwork.Loader;
using ML.NeuralNetwork.LossFunctions;
using ML.NeuralNetwork.Optimizers;
using NUnit.Framework;
using NUnit.Framework.Legacy;

using static ML.NeuralNetwork.Loader.IDataLoader;

namespace ML.NeuralNetwork.Configs;

public static class Chess
{
    public static double WDL = 0.75;
    private static int Scale => 400;
    private static int InputLayerSize => 768;
    private static int OutputLayerSize => 1;
    
    private static double Sigmoid_fn(double x) => 1.0 / (1.0 + Math.Exp(-x));
    
    private static readonly Dictionary<char, int> PIECE_TYPES = new()
    {
        {'p', 0},
        {'n', 1},
        {'b', 2},
        {'r', 3},
        {'q', 4},
        {'k', 5}
    };
    
    public static void ParseChessPosition(LoadContext context)
    {
        var fen = context.Line;
        var index = 0;
        var square = 0;
        
        for (; !char.IsSeparator(fen[index]); index++)
        {
            var c = fen[index];
            if (c == '/')
                continue;
            
            var cInt = c - '0';
            if (cInt is >= 1 and <= 8)
            {
                square += cInt;
                continue;
            }
            
            var pieceColor = char.IsUpper(c) ? 0 : 1; // WHITE, BLACK;
            var pieceType = PIECE_TYPES[char.ToLower(c)];
            var indexToSet = GetIndex(pieceColor, pieceType, square);
            context.Input[indexToSet] = 1;
            square++;
        }
        
        while (fen[index++] != '|'){}
        index++;
        
        Span<char> num = stackalloc char[6];
        var size = 0;
        while (!char.IsSeparator(fen[index]))
            num[size++] = fen[index++];
        var cp = double.Parse(num);
            
        while (fen[index++] != '|'){}
        index++;
        
        double wdl;
        if (fen[index] == '0')
            wdl = fen[index + 2] == '5' ? 0.5 : 0;
        else
            wdl = 1;
        
        context.Output[0] = Sigmoid_fn(cp / Scale) * (1.0 - WDL) + wdl * WDL;
        return;
        static int GetIndex(int color, int piece, int square) => (color * 6 + piece) * 64 + square;
    }
    
    public static NeuralNetwork Create(int hiddenLayerSize, string dataPath, string? netPath = null)
    {
        var nn = new NeuralNetwork()
            .AddInputLayer(InputLayerSize)
            .AddLayer(hiddenLayerSize, typeof(RELU))
            .AddLayer(1, typeof(Sigmoid))
            .SetDataLoader(new DataLoader(dataPath, ParseChessPosition, InputLayerSize, OutputLayerSize))
            .SetLossFunction(typeof(MSE))
            .SetOptimizer(new AdamW
            {
                Configuration = new AdamW.Config()
            })
            .UseQuantization([255, 64])
            .Build();
        
        if (netPath is not null)  
            nn.Load(netPath);
        else
            nn.InitializeRandom();
        
        return nn;
    }
}

[TestFixture]
public class ChessTest
{
    [Test]
    public void ParseFenTest()
    {
        var context = new LoadContext("2R5/2K5/8/8/1k6/5Q2/8/8 w - - 3 72 | -31860 | 0.0", new double[768], new double[1]);
        var expectedInputFeatures = new double[768];
        expectedInputFeatures[194] = 1; // R
        expectedInputFeatures[330] = 1; // K
        expectedInputFeatures[301] = 1; // Q
        expectedInputFeatures[737] = 1; // k
        
        Chess.ParseChessPosition(context);
        CollectionAssert.AreEqual(expectedInputFeatures, context.Input);
    }

    [Test]
    public void ParseFenTest2()
    {
        var context = new IDataLoader.LoadContext("1rb1kb1r/2ppqp1p/1pn2np1/p3p3/2P1P1P1/1P1PBN1B/P4P1P/RN1QK2R w KQk - 2 9 | 130 | 1.0", new double[768], new double[1]);
        var expectedInputFeatures = new double[768];
        // White pawns.
        SetFeature(0,0,34);
        SetFeature(0,0,36);
        SetFeature(0,0,38);
        SetFeature(0,0,41);
        SetFeature(0,0,43);
        SetFeature(0,0,48);
        SetFeature(0,0,53);
        SetFeature(0,0,55);
        // White knights.
        SetFeature(0,1,45);
        SetFeature(0,1,57);
        // White bishops.
        SetFeature(0,2,44);
        SetFeature(0,2,47);
        // White rooks.
        SetFeature(0,3,56);
        SetFeature(0,3,63);
        // White queen.
        SetFeature(0,4,59);
        // White king.
        SetFeature(0,5,60);
        
        // Black pawns.
        SetFeature(1,0,10);
        SetFeature(1,0,11);
        SetFeature(1,0,13);
        SetFeature(1,0,15);
        SetFeature(1,0,17);
        SetFeature(1,0,22);
        SetFeature(1,0,24);
        SetFeature(1,0,28);
        // Black knights.
        SetFeature(1,1,18);
        SetFeature(1,1,21);
        // Black bishops.
        SetFeature(1,2,2);
        SetFeature(1,2,5);
        // Black rooks.
        SetFeature(1,3,1);
        SetFeature(1,3,7);
        // Black queen.
        SetFeature(1,4,12);
        // Black king.
        SetFeature(1,5,4);
        
        Chess.ParseChessPosition(context);
        CollectionAssert.AreEqual(expectedInputFeatures, context.Input);
        
        return; 
        void SetFeature(int color, int piece, int square) => expectedInputFeatures[GetIndex(color, piece, square)] = 1;
        static int GetIndex(int color, int piece, int square) => (color * 6 + piece) * 64 + square;
    }
}