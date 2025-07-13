using System.Collections.Concurrent;
using static ML.NeuralNetwork.NeuralNetworkHelper;

namespace ML.NeuralNetwork;

public static class NeuralNetworkArrayPool
{
    private static readonly ConcurrentQueue<double[]>[] _pool = [new (), new ()];
    
    public static double[] Rent(int size, bool input)
    {
        var poolIndex = Index(input);
        var queue = _pool[poolIndex];
        if (queue.IsEmpty) 
            return new double[size];
        
        return queue.TryDequeue(out var item) ? item : new double[size];
    }

    public static void Return(TrainingItem trainingItem)
    {
        Clear(trainingItem.Input);
        _pool[Index(true)].Enqueue(trainingItem.Input);
        
        Clear(trainingItem.Expected);
        _pool[Index(false)].Enqueue(trainingItem.Expected);
    }

    private static void Clear(double[] array)
    {
        for (var i = 0; i < array.Length; i++)
            array[i] = 0;
    }
    
    private static int Index(bool input) => input ? 1 : 0;
}