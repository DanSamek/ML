namespace ML.NeuralNetwork;

/// <summary>
/// Used activation functions.
/// </summary>
public class ActivationFunctions
{
    public static double Sigmoid(double current) => 1 / (1 + Math.Exp(-current));
    public static double RELU(double current) => double.Max(0, current);
    
    public static int CreluMax { get; set; } = 256;
    public static double CRELU(double current) => double.Clamp(current, 0, CreluMax);
    
    
    public static int ScreluMax { get; set; } = 256;
    public static double SCRELU(double current) => Math.Pow(double.Clamp(current, 0, ScreluMax), 2);
}